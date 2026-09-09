"""The loop CLOSES: an optimizer step through the kernel moves the policy.

The equivalence suite proves the kernel computes the same NUMBER as the
audited oracle. That is necessary and not sufficient: a scalar can agree to
1e-6 and still carry a gradient that points nowhere useful. This module
measures the remaining claim -- that stepping an optimizer on the kernel's
output changes a real parameter, in the DIRECTION the advantages ask for.

Direction, not descent, is the assertion. "The loss went down" is nearly
vacuous: any differentiable scalar descends under its own gradient, including
one computed with the sign flipped or the advantages shuffled. The
falsifiable claim is that a row with a POSITIVE advantage becomes more likely
and a row with a NEGATIVE advantage becomes less likely -- and that flipping
the advantages flips that outcome.

WHAT IS CLAIMED: on the fixed setup below, training through
``TensorPolicyLoss`` moves each row's mean log-probability in the sign of its
advantage, and negating the advantages negates every one of those movements.

WHAT IS NOT CLAIMED: convergence, sample efficiency, or that any of this
holds for a real language model on a real corpus -- this is a tiny linear
policy on a fixed batch, and it measures the gradient path, not the recipe.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from foundationscale.rl.group_policy_objectives import DrGRPOLoss  # noqa: E402
from foundationscale.rl.torch_backend import TensorPolicyLoss  # noqa: E402

_ROWS = 4
_TOKENS = 5
_STEPS = 60


class _TinyPolicy(torch.nn.Module):
    """A minimal differentiable stand-in for a policy's log-probabilities.

    One free parameter per row, broadcast over tokens, passed through
    ``-softplus`` so every output is a valid (negative) log-probability. The
    point is a REAL autograd path with real parameters, not a language model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(_ROWS, _TOKENS))

    def forward(self) -> torch.Tensor:
        return -torch.nn.functional.softplus(self.logits)


def _train(advantages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Train the tiny policy and return (start, end) per-row mean logprobs."""
    torch.manual_seed(0)
    policy = _TinyPolicy()
    objective = DrGRPOLoss(group_size=2)
    kernel = TensorPolicyLoss(objective)
    mask = torch.ones((_ROWS, _TOKENS), dtype=torch.float32)

    with torch.no_grad():
        start = policy().mean(dim=-1).clone()

    # old_logprobs is the frozen sampling policy: fixed at the INITIAL
    # readings, so the importance ratio starts at exactly 1 and moves as
    # the policy moves, which is what makes the clip reachable at all.
    old = policy().detach().clone()
    optimiser = torch.optim.SGD(policy.parameters(), lr=0.5)
    for _step in range(_STEPS):
        optimiser.zero_grad()
        loss = kernel(
            current_logprobs=policy(),
            old_logprobs=old,
            advantages=advantages,
            mask=mask,
        )
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        end = policy().mean(dim=-1).clone()
    return start, end


def test_optimiser_step_moves_each_row_in_the_sign_of_its_advantage() -> None:
    advantages = torch.tensor([1.0, -1.0, 2.0, -0.5])
    start, end = _train(advantages)
    delta = end - start
    assert bool((delta != 0.0).all()), (
        f"0 of {_ROWS} rows moved at all: the optimizer received a scalar "
        f"with no usable gradient, which is the #362 defect restated"
    )
    for row in range(_ROWS):
        want = float(advantages[row])
        got = float(delta[row])
        assert (got > 0) == (want > 0), (
            f"row {row} has advantage {want!r} but its mean log-probability "
            f"moved by {got!r}; a positively-advantaged row must become MORE "
            f"likely and a negatively-advantaged row LESS likely"
        )


def test_negating_the_advantages_negates_every_movement() -> None:
    """The control. Without it, the test above cannot separate a real
    gradient from any monotone function of the parameters."""
    advantages = torch.tensor([1.0, -1.0, 2.0, -0.5])
    start, end = _train(advantages)
    flipped_start, flipped_end = _train(-advantages)
    assert bool(torch.allclose(start, flipped_start)), (
        "the two arms did not start from the same parameters; the comparison "
        "below would then confound initialisation with the advantage sign"
    )
    delta = end - start
    flipped_delta = flipped_end - flipped_start
    for row in range(_ROWS):
        assert (float(delta[row]) > 0) != (float(flipped_delta[row]) > 0), (
            f"row {row} moved {float(delta[row])!r} with the advantages and "
            f"{float(flipped_delta[row])!r} against them; negating the "
            f"advantage must reverse the direction, or the movement is not "
            f"being driven by the advantage at all"
        )


def test_parameters_actually_changed() -> None:
    # The weakest claim, stated separately so its failure is unambiguous:
    # if the parameters are bit-identical after 60 steps, nothing above
    # measured anything, however green it reads.
    torch.manual_seed(0)
    before = _TinyPolicy().logits.detach().clone()
    advantages = torch.tensor([1.0, -1.0, 2.0, -0.5])
    torch.manual_seed(0)
    policy = _TinyPolicy()
    kernel = TensorPolicyLoss(DrGRPOLoss(group_size=2))
    mask = torch.ones((_ROWS, _TOKENS), dtype=torch.float32)
    old = policy().detach().clone()
    optimiser = torch.optim.SGD(policy.parameters(), lr=0.5)
    for _step in range(_STEPS):
        optimiser.zero_grad()
        kernel(
            current_logprobs=policy(),
            old_logprobs=old,
            advantages=advantages,
            mask=mask,
        ).backward()
        optimiser.step()
    moved = int((policy.logits.detach() != before).sum())
    assert moved == _ROWS * _TOKENS, (
        f"{moved} of {_ROWS * _TOKENS} parameters changed after {_STEPS} "
        f"optimizer steps; every supervised position should have moved"
    )
