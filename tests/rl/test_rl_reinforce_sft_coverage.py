# SPDX-License-Identifier: Apache-2.0
"""Coverage of the refusal / observability tails in the RL tensor plane.

WHAT IS CLAIMED: every test here executes a previously-uncovered refusal or
tail branch in ``trainer.py`` / ``torch_backend.py`` and asserts the
behaviour that branch owes -- the exception type together with a
distinctive phrase of the real message, or the value the branch must
return. The REINFORCE / SFT tails are driven directly with hand-built
tensors and stub objectives, which the module itself invites for branches
unreachable from ``run()``.

WHAT IS NOT CLAIMED: convergence, real-model loading, or any property of
the ships' generation quality. The fake host comes from
test_trainer_group_family (imported as a top-level module: tests/rl has no
__init__.py, so ``tests.rl.``-style imports cannot work there).
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any

import pytest

torch = pytest.importorskip("torch")

from test_trainer_group_family import (  # noqa: E402 (sibling test module, no package)
    _config,
    _ConstantReward,
    _install_fake_host,
)

import foundationscale.rl.trainer as trainer_module  # noqa: E402 (torch importorskip first)
from foundationscale.rl.interfaces import BatchRefusal  # noqa: E402 (torch importorskip first)
from foundationscale.rl.torch_backend import (  # noqa: E402 (torch importorskip first)
    TensorMaskedSFTLoss,
    TensorPolicyLoss,
    TensorREINFORCELoss,
)
from foundationscale.rl.trainer import (  # noqa: E402 (torch importorskip first)
    RLTrainer,
    TrainerRefusal,
    _micro_batched_backward,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable


# --- shared stubs ------------------------------------------------------------


class _RecordingOptimizer:
    """Counts zero_grad/step calls: each tail must step exactly once."""

    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class _TokenPolicyStub:
    """Minimal token-scope objective axes for the ratio kernel."""

    ratio_scope: str = "token"
    clip_bounds: tuple[float, float] = (0.8, 1.2)
    reduction: str = "token_mean"

    def declaration(self) -> Any:
        return types.SimpleNamespace(components=("policy",))


class _ReinforceBaselineStub:
    """Declared EMA state for the reinforce_baseline tail."""

    baseline_momentum: float = 0.99
    baseline_metric_name: str = "reinforce_baseline_frac_above"

    def declaration(self) -> Any:
        return types.SimpleNamespace(components=("surrogate",))


class _ReinforcePPStub:
    """Declared k1 fold plus token-clip axes for the reinforce_pp tail."""

    kl_beta: float = 0.04
    ratio_scope: str = "token"
    clip_bounds: tuple[float, float] = (0.8, 1.2)
    reduction: str = "token_mean"

    def declaration(self) -> Any:
        return types.SimpleNamespace(components=("policy",))


class _ReferenceOnlyBinding:
    """A binding exposing NO ``_objective`` but declaring a reference need."""

    def requirements(self) -> Any:
        return types.SimpleNamespace(requires={"reference_policy": True})


def _micro_planes() -> tuple[
    torch.nn.Parameter,
    torch.Tensor,
    tuple[tuple[int, int], ...],
    Callable[[int, int], torch.Tensor],
]:
    """One parameter-backed forward plus the detached batch-sized leaf.

    ``forward_slice(start, end)`` re-creates a fresh narrow graph each call,
    which is exactly what ``_micro_batched_backward`` chains leaf gradients
    through; ``leaf`` is the detached reading the tails price against.
    """
    param = torch.nn.Parameter(torch.full((3, 2), -0.5))

    def forward_slice(start: int, end: int) -> torch.Tensor:
        return (param * 2.0).narrow(0, start, end - start)

    row_slices = ((0, 2), (2, 3))
    leaf = torch.cat([forward_slice(0, 2), forward_slice(2, 3)], dim=0).detach()
    leaf.requires_grad_(True)
    return param, leaf, row_slices, forward_slice


# --- trainer.py line 152: _micro_batched_backward's missing-gradient guard --


def test_micro_batched_backward_refuses_when_the_leaf_took_no_gradient() -> None:
    # A loss priced off something OTHER than the detached leaf leaves
    # leaf.grad None after backward; chaining slice backward cannot proceed.
    leaf = torch.zeros((2, 2), requires_grad=True)
    source = torch.tensor(0.5, requires_grad=True)
    with pytest.raises(RuntimeError, match="produced no gradient"):
        _micro_batched_backward(
            loss_tensor=source * 2.0,
            current_logprobs=leaf,
            row_slices=((0, 1),),
            forward_slice=lambda start, end: leaf.narrow(0, start, end - start),
        )


# --- trainer.py line 432: reference-declared binding with no objective ------


def test_resolve_objective_refuses_binding_declaring_reference_without_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer_module, "lookup_algorithm", lambda name: _ReferenceOnlyBinding())
    trainer = RLTrainer(_config("imaginary_ref_algo"))
    with pytest.raises(TrainerRefusal, match=r"requires\.reference_policy=True"):
        trainer._resolve_objective()


# --- trainer.py lines 942, 943, 947, 1121: micro-batched PPO-clip planes ----


def test_rloo_with_logprob_micro_batching_prices_one_logical_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # old/current planes become concatenations of no-grad slices and the
    # backward is delivered through the detached leaf, one slice at a time.
    _install_fake_host(monkeypatch)
    reports = RLTrainer(_config("rloo", logprob_micro_batch=1)).run()
    assert len(reports) == 2
    for report in reports:
        assert report.rows > 0


# --- trainer.py line 1235: truncated flat-group statement in _sft_tail ------


def test_sft_tail_states_truncated_flat_group_count_as_a_count(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Ten flat groups exceed _MAX_GROUPS_REPORTED (8): the tail names the
    # remainder as "(+2 more group(s))" instead of printing all ten.
    _install_fake_host(monkeypatch)
    monkeypatch.setattr(trainer_module, "MCQLetterReward", lambda **kwargs: _ConstantReward())
    with pytest.raises(TrainerRefusal, match="vacuous"):
        RLTrainer(_config("raft", prompts_per_step=10, max_steps=1)).run()
    err = capsys.readouterr().err
    assert "10 of 10 group(s) dropped from the SFT tail" in err
    assert "(+2 more group(s))" in err
    assert "every group is flat" in err


# --- trainer.py line 1364: reinforce_baseline needs a declared momentum -----


def test_reinforce_baseline_tail_refuses_an_undeclared_momentum() -> None:
    trainer = RLTrainer(_config("reinforce_baseline"))
    with pytest.raises(TrainerRefusal, match=r"declares no baseline_momentum"):
        trainer._reinforce_baseline_tail(
            step=0,
            scores=[1.0],
            kept_current=torch.zeros((1, 1), requires_grad=True),
            kept_mask=torch.ones((1, 1)),
            current_logprobs=torch.zeros((1, 1), requires_grad=True),
            row_slices=((0, 1),),
            forward_slice=lambda start, end: torch.zeros((end - start, 1)),
            use_logprob_micro_batching=False,
            objective=types.SimpleNamespace(),
            optimizer=_RecordingOptimizer(),
        )


# --- trainer.py lines 1379, 1386: zero-stimulus REINFORCE batch is UNMEASURED


def test_reinforce_baseline_tail_flat_returns_are_unmeasured(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trainer = RLTrainer(_config("reinforce_baseline"))
    result = trainer._reinforce_baseline_tail(
        step=3,
        scores=[1.0, 1.0],
        kept_current=torch.zeros((2, 1), requires_grad=True),
        kept_mask=torch.ones((2, 1)),
        current_logprobs=torch.zeros((2, 1), requires_grad=True),
        row_slices=((0, 2),),
        forward_slice=lambda start, end: torch.zeros((end - start, 1)),
        use_logprob_micro_batching=False,
        objective=_ReinforceBaselineStub(),
        optimizer=_RecordingOptimizer(),
    )
    assert result is None
    assert "every kept return equals the baseline" in capsys.readouterr().err
    assert trainer._reinforce_baseline is None, "an unmeasured step must not seed the EMA"


# --- trainer.py line 1396: reinforce_baseline micro-batched backward --------


def test_reinforce_baseline_tail_micro_batch_delivers_gradient_through_slices() -> None:
    param, leaf, row_slices, forward_slice = _micro_planes()
    optimizer = _RecordingOptimizer()
    trainer = RLTrainer(_config("reinforce_baseline"))
    report = trainer._reinforce_baseline_tail(
        step=0,
        scores=[1.0, 0.0, 1.0],
        kept_current=leaf,
        kept_mask=torch.ones((3, 2)),
        current_logprobs=leaf,
        row_slices=row_slices,
        forward_slice=forward_slice,
        use_logprob_micro_batching=True,
        objective=_ReinforceBaselineStub(),
        optimizer=optimizer,
    )
    assert report is not None
    assert report.rows == 3
    assert optimizer.zero_grad_calls == 1
    assert optimizer.step_calls == 1
    assert param.grad is not None and bool(param.grad.abs().any())
    assert trainer._reinforce_baseline == pytest.approx(2.0 / 3.0)


# --- trainer.py line 1480: reinforce_pp needs the reference plane -----------


def test_reinforce_pp_tail_refuses_a_missing_reference_plane() -> None:
    trainer = RLTrainer(_config("reinforce_pp"))
    with pytest.raises(TrainerRefusal, match="reinforce_pp folds reference log-probabilities"):
        trainer._reinforce_pp_tail(
            step=0,
            scores=[1.0],
            kept_current=torch.zeros((1, 1), requires_grad=True),
            kept_old=torch.zeros((1, 1)),
            kept_mask=torch.ones((1, 1)),
            kept_ref=None,
            current_logprobs=torch.zeros((1, 1), requires_grad=True),
            row_slices=((0, 1),),
            forward_slice=lambda start, end: torch.zeros((end - start, 1)),
            use_logprob_micro_batching=False,
            objective=_ReinforcePPStub(),
            loss_fn=TensorPolicyLoss(objective=_ReinforcePPStub()),
            optimizer=_RecordingOptimizer(),
        )


# --- trainer.py line 1488: reinforce_pp needs a declared kl_beta ------------


def test_reinforce_pp_tail_refuses_an_undeclared_kl_beta() -> None:
    trainer = RLTrainer(_config("reinforce_pp"))
    with pytest.raises(TrainerRefusal, match=r"declares no kl_beta"):
        trainer._reinforce_pp_tail(
            step=0,
            scores=[1.0],
            kept_current=torch.zeros((1, 1), requires_grad=True),
            kept_old=torch.zeros((1, 1)),
            kept_mask=torch.ones((1, 1)),
            kept_ref=torch.zeros((1, 1)),
            current_logprobs=torch.zeros((1, 1), requires_grad=True),
            row_slices=((0, 1),),
            forward_slice=lambda start, end: torch.zeros((end - start, 1)),
            use_logprob_micro_batching=False,
            objective=types.SimpleNamespace(),
            loss_fn=TensorPolicyLoss(objective=_ReinforcePPStub()),
            optimizer=_RecordingOptimizer(),
        )


# --- trainer.py lines 1502, 1510: zero-spread z-scores are UNMEASURED -------


def test_reinforce_pp_tail_zero_spread_is_unmeasured(
    capsys: pytest.CaptureFixture[str],
) -> None:
    trainer = RLTrainer(_config("reinforce_pp"))
    current = torch.full((2, 2), -1.0, requires_grad=True)
    result = trainer._reinforce_pp_tail(
        step=5,
        scores=[0.5, 0.5],
        kept_current=current,
        kept_old=current.detach().clone(),
        kept_mask=torch.ones((2, 2)),
        kept_ref=current.detach().clone(),
        current_logprobs=current,
        row_slices=((0, 2),),
        forward_slice=lambda start, end: current.narrow(0, start, end - start),
        use_logprob_micro_batching=False,
        objective=_ReinforcePPStub(),
        loss_fn=TensorPolicyLoss(objective=_ReinforcePPStub()),
        optimizer=_RecordingOptimizer(),
    )
    assert result is None
    assert "zero spread" in capsys.readouterr().err


# --- trainer.py line 1524: reinforce_pp micro-batched backward --------------


def test_reinforce_pp_tail_micro_batch_delivers_gradient_through_slices() -> None:
    param, leaf, row_slices, forward_slice = _micro_planes()
    optimizer = _RecordingOptimizer()
    trainer = RLTrainer(_config("reinforce_pp"))
    report = trainer._reinforce_pp_tail(
        step=0,
        scores=[1.0, 0.0, 1.0],
        kept_current=leaf,
        kept_old=leaf.detach().clone(),
        kept_mask=torch.ones((3, 2)),
        kept_ref=torch.full((3, 2), -1.25),
        current_logprobs=leaf,
        row_slices=row_slices,
        forward_slice=forward_slice,
        use_logprob_micro_batching=True,
        objective=_ReinforcePPStub(),
        loss_fn=TensorPolicyLoss(objective=_ReinforcePPStub()),
        optimizer=optimizer,
    )
    assert report is not None
    assert report.rows == 3
    assert optimizer.step_calls == 1
    assert param.grad is not None and bool(param.grad.abs().any())


# --- torch_backend.py TensorPolicyLoss guards: lines 59, 160, 191, 200, 208


def test_policy_kernel_refuses_non_tensor_current_logprobs() -> None:
    with pytest.raises(BatchRefusal, match="not torch.Tensor"):
        TensorPolicyLoss(_TokenPolicyStub())(
            current_logprobs="not-a-tensor",  # type: ignore[arg-type]
            old_logprobs=torch.zeros((1, 2)),
            advantages=torch.ones(1),
            mask=torch.ones((1, 2)),
        )


def test_policy_kernel_refuses_a_non_2d_current_plane() -> None:
    with pytest.raises(BatchRefusal, match="exactly 2 dims"):
        TensorPolicyLoss(_TokenPolicyStub())(
            current_logprobs=torch.zeros((2, 3, 1), requires_grad=True),
            old_logprobs=torch.zeros((2, 3)),
            advantages=torch.ones(2),
            mask=torch.ones((2, 3)),
        )


def test_policy_kernel_refuses_a_row_mismatched_advantage_vector() -> None:
    with pytest.raises(BatchRefusal, match=r"advantages has 5 rows but"):
        TensorPolicyLoss(_TokenPolicyStub())(
            current_logprobs=torch.zeros((2, 3), requires_grad=True),
            old_logprobs=torch.zeros((2, 3)),
            advantages=torch.ones(5),
            mask=torch.ones((2, 3)),
        )


def test_policy_kernel_refuses_3d_advantages() -> None:
    with pytest.raises(BatchRefusal, match=r"a \(rows,\) vector or a"):
        TensorPolicyLoss(_TokenPolicyStub())(
            current_logprobs=torch.zeros((2, 3), requires_grad=True),
            old_logprobs=torch.zeros((2, 3)),
            advantages=torch.zeros((2, 3, 1)),
            mask=torch.ones((2, 3)),
        )


def test_policy_kernel_refuses_a_fractional_mask() -> None:
    mask = torch.ones((2, 3))
    mask[0, 0] = 0.5
    with pytest.raises(BatchRefusal, match=r"1 of 6 mask entries are neither 0 nor 1"):
        TensorPolicyLoss(_TokenPolicyStub())(
            current_logprobs=torch.zeros((2, 3), requires_grad=True),
            old_logprobs=torch.zeros((2, 3)),
            advantages=torch.ones(2),
            mask=mask,
        )


# --- torch_backend.py TensorMaskedSFTLoss guards: 391, 399, 411, 419, 430, 443


def test_sft_kernel_refuses_non_tensor_current_logprobs() -> None:
    with pytest.raises(BatchRefusal, match="not torch.Tensor"):
        TensorMaskedSFTLoss(objective=types.SimpleNamespace())(
            current_logprobs="not-a-tensor",  # type: ignore[arg-type]
            mask=torch.ones((1, 2)),
        )


def test_sft_kernel_refuses_a_gradient_free_current_plane() -> None:
    with pytest.raises(BatchRefusal, match="requires_grad is False"):
        TensorMaskedSFTLoss(objective=types.SimpleNamespace())(
            current_logprobs=torch.zeros((2, 3)),
            mask=torch.ones((2, 3)),
        )


def test_sft_kernel_refuses_a_fractional_mask() -> None:
    mask = torch.ones((2, 3))
    mask[1, 2] = 0.25
    with pytest.raises(BatchRefusal, match=r"mask entries are neither 0 nor 1"):
        TensorMaskedSFTLoss(objective=types.SimpleNamespace())(
            current_logprobs=torch.zeros((2, 3), requires_grad=True),
            mask=mask,
        )


def test_sft_kernel_refuses_a_fully_masked_batch() -> None:
    with pytest.raises(BatchRefusal, match=r"0 of 4 mask entries are supervised"):
        TensorMaskedSFTLoss(objective=types.SimpleNamespace())(
            current_logprobs=torch.zeros((2, 2), requires_grad=True),
            mask=torch.zeros((2, 2)),
        )


def test_sft_kernel_refuses_a_row_with_no_supervision() -> None:
    mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    with pytest.raises(BatchRefusal, match=r"1 of 2 rows carry 0 supervised tokens"):
        TensorMaskedSFTLoss(objective=types.SimpleNamespace())(
            current_logprobs=torch.zeros((2, 2), requires_grad=True),
            mask=mask,
        )


def test_sft_kernel_refuses_a_non_finite_loss_from_an_infinite_weight() -> None:
    with pytest.raises(BatchRefusal, match="non-finite loss"):
        TensorMaskedSFTLoss(objective=types.SimpleNamespace(weight=float("inf")))(
            current_logprobs=torch.full((2, 2), -1.0, requires_grad=True),
            mask=torch.ones((2, 2)),
        )


# --- torch_backend.py TensorREINFORCELoss guards: 502, 510, 522-556 ---------


def test_reinforce_kernel_refuses_non_tensor_current_logprobs() -> None:
    with pytest.raises(BatchRefusal, match="not torch.Tensor"):
        TensorREINFORCELoss(objective=types.SimpleNamespace())(
            current_logprobs="not-a-tensor",  # type: ignore[arg-type]
            advantages=torch.ones(1),
            mask=torch.ones((1, 2)),
        )


def test_reinforce_kernel_refuses_a_gradient_free_current_plane() -> None:
    with pytest.raises(BatchRefusal, match="requires_grad is False"):
        TensorREINFORCELoss(objective=types.SimpleNamespace())(
            current_logprobs=torch.zeros((2, 3)),
            advantages=torch.ones(2),
            mask=torch.ones((2, 3)),
        )


def test_reinforce_kernel_refuses_non_vector_advantages() -> None:
    with pytest.raises(BatchRefusal, match=r"a \(rows,\) vector of one scalar weight"):
        TensorREINFORCELoss(objective=types.SimpleNamespace())(
            current_logprobs=torch.zeros((2, 3), requires_grad=True),
            advantages=torch.ones((2, 3)),
            mask=torch.ones((2, 3)),
        )


def test_reinforce_kernel_refuses_row_mismatched_advantages() -> None:
    with pytest.raises(
        BatchRefusal, match=r"advantages has 3 rows but current_logprobs has 2 rows"
    ):
        TensorREINFORCELoss(objective=types.SimpleNamespace())(
            current_logprobs=torch.zeros((2, 3), requires_grad=True),
            advantages=torch.ones(3),
            mask=torch.ones((2, 3)),
        )


def test_reinforce_kernel_refuses_a_fractional_mask() -> None:
    mask = torch.ones((2, 3))
    mask[0, 1] = 0.5
    with pytest.raises(BatchRefusal, match=r"1 of 6 mask entries are neither 0 nor 1"):
        TensorREINFORCELoss(objective=types.SimpleNamespace())(
            current_logprobs=torch.zeros((2, 3), requires_grad=True),
            advantages=torch.ones(2),
            mask=mask,
        )


def test_reinforce_kernel_refuses_a_fully_masked_batch() -> None:
    with pytest.raises(BatchRefusal, match=r"0 of \d+ mask entries are supervised"):
        TensorREINFORCELoss(objective=types.SimpleNamespace())(
            current_logprobs=torch.zeros((2, 3), requires_grad=True),
            advantages=torch.ones(2),
            mask=torch.zeros((2, 3)),
        )


def test_reinforce_kernel_refuses_a_non_finite_loss_from_an_infinite_weight() -> None:
    with pytest.raises(BatchRefusal, match="non-finite loss"):
        TensorREINFORCELoss(objective=types.SimpleNamespace(weight=float("inf")))(
            current_logprobs=torch.full((2, 3), -1.0, requires_grad=True),
            advantages=torch.ones(2),
            mask=torch.ones((2, 3)),
        )
