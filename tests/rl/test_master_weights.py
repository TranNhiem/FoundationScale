"""MasterWeightOptimizer: does it recover the update bf16 discards? (#369)

The claim under test is NOT "the wrapper runs". It is the numerical one that
[[#368]] measured on a GB200 and that #369 found the shipped trainer violates:
bf16 parameters stepped directly by AdamW at a small lr do not move, because
each update is below the bf16 ulp and is DISCARDED rather than attenuated, so
nothing accumulates across steps.

The measured reference (gemma-4-E4B, 679M gradient-carrying entries, lr=1e-6,
50 steps): bf16 direct stays FLAT at 0.98% of entries moved -- the same after
50 steps as after 1 -- while host-fp32 masters reach 20.58%, and device peak
DROPS 109.51 -> 68.08 GiB because the Adam state leaves the GPU.

These legs reproduce the separation on CPU with a tiny model, which is enough
to pin the mechanism: both arms get the same seed, the same weights and the
same gradients, so the optimiser is the only independent variable. They do NOT
claim the GB200 percentages -- those depend on the model and are recorded in
the task, not asserted here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from foundationscale.rl.trainer import MasterWeightOptimizer  # noqa: E402

LR = 1e-6
STEPS = 40


def _run(use_masters: bool) -> tuple[int, int]:
    """Train one tiny bf16 model and return (entries moved, entries total)."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64),
        torch.nn.Linear(64, 8),
    ).to(torch.bfloat16)
    initial = [p.detach().clone() for p in model.parameters()]
    optimizer = (
        MasterWeightOptimizer(list(model.parameters()), lr=LR)
        if use_masters
        else torch.optim.AdamW(model.parameters(), lr=LR)
    )
    torch.manual_seed(1)
    inputs = torch.randn(32, 64, dtype=torch.bfloat16)
    targets = torch.randn(32, 8, dtype=torch.bfloat16)
    for _ in range(STEPS):
        loss = torch.nn.functional.mse_loss(model(inputs).float(), targets.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    moved = sum(
        int((p.detach() != i).sum()) for p, i in zip(model.parameters(), initial, strict=True)
    )
    return moved, sum(p.numel() for p in model.parameters())


def test_masters_move_far_more_of_the_model_than_direct_bf16() -> None:
    # The load-bearing leg. A wrapper that merely ran would pass a smoke test
    # and still train nothing, which is precisely the defect #369 records, so
    # the assertion is on the SEPARATION between the arms.
    direct_moved, total = _run(use_masters=False)
    master_moved, total_again = _run(use_masters=True)
    assert total == total_again
    assert master_moved > direct_moved * 5, (
        f"masters moved {master_moved}/{total} vs direct {direct_moved}/{total}; "
        "the fp32 accumulation is not reaching the device parameters"
    )


def test_direct_bf16_leaves_almost_the_whole_model_untouched() -> None:
    # Pins the DEFECT itself, not just the fix. If a future torch makes bf16
    # AdamW accumulate sub-ulp updates, this leg fails and tells us the
    # premise changed -- rather than the fix silently becoming pointless.
    moved, total = _run(use_masters=False)
    assert moved < total * 0.10, (
        f"direct bf16 moved {moved}/{total}; the sub-ulp-discard premise "
        "behind #369 no longer holds on this toolchain"
    )


def test_zero_grad_clears_both_planes() -> None:
    # zero_grad must clear the DEVICE grads (what the loop writes) and the
    # MASTER grads (what AdamW reads). Leaving either stale double-counts.
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 4).to(torch.bfloat16)
    optimizer = MasterWeightOptimizer(list(model.parameters()), lr=LR)
    model(torch.randn(2, 8, dtype=torch.bfloat16)).float().sum().backward()
    optimizer.step()
    assert any(p.grad is not None for p in model.parameters())
    optimizer.zero_grad()
    assert all(p.grad is None for p in model.parameters())
    assert all(m.grad is None for m in optimizer.masters)


def test_master_count_matches_trainable_params() -> None:
    # The strict=True zips depend on this. A frozen parameter must not acquire
    # a master, or every later pairing is off by one and trains the wrong
    # tensor -- silently, with a finite loss.
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    model[0].weight.requires_grad_(False)
    optimizer = MasterWeightOptimizer(list(model.parameters()), lr=LR)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(optimizer.masters) == len(trainable)
    assert len(optimizer.device_params) == len(trainable)
