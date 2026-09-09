"""Tensor-plane agreement with the audited pure-python ORACLE.

The shipped objectives in ``losses.py`` / ``group_policy_objectives.py``
are the oracle: audited, torch-free, and unchanged. ``TensorPolicyLoss``
is trustworthy exactly insofar as it reproduces them. On any disagreement
the TENSOR path is what is wrong, never the oracle.

For each of GRPO, GSPO, Dr.GRPO and DAPO a fixed pseudo-random batch is
priced by the oracle's ``forward_fn`` pipeline, the estimator's own used
rows and per-token weights are re-fed to the tensor kernel, and the two
losses must agree to ``abs(a - b) <= 1e-6``. Gradient flow and every
kernel-side refusal are tested beside the equivalences.

WHAT IS CLAIMED: the tensor kernel agrees with the oracle arithmetic on
the batches below, across every declared combination of the four axes the
oracle family spans. WHAT IS NOT CLAIMED: any convergence, benchmark, or
paper-equivalence property, and any guarantee beyond the batches measured.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from foundationscale.rl.group_policy_objectives import (  # noqa: E402
    DAPOLoss,
    DrGRPOLoss,
    GSPOLoss,
)
from foundationscale.rl.grpo import GRPOPolicyLoss  # noqa: E402
from foundationscale.rl.interfaces import (  # noqa: E402
    BatchRefusal,
    ExperienceBatch,
)
from foundationscale.rl.torch_backend import TensorPolicyLoss  # noqa: E402

_ROWS = 4
_TOKENS = 6


def _fixed_batch() -> tuple[
    tuple[str, ...],
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    """One fixed pseudo-random batch, identical for every test below."""
    rng = random.Random(0)
    prompt_ids = ("alpha", "alpha", "beta", "beta")
    rewards = (1.0, 0.0, 2.0, 0.5)
    current = tuple(tuple(rng.uniform(-3.0, -0.5) for _ in range(_TOKENS)) for _ in range(_ROWS))
    # The 0.30 half-width is load-bearing, not cosmetic. It puts ratios in
    # [0.756, 1.349], which straddles GRPO's (0.8, 1.2) AND DAPO's asymmetric
    # (0.8, 1.28) -- including tokens in (1.2, 1.28], the only band where the
    # two disagree. At the original 0.05 every ratio sat inside every bound,
    # so torch.clamp was the identity everywhere except GSPO's near-1 window:
    # a kernel ignoring clip_bounds entirely passed all four equivalences,
    # and DAPO's one distinguishing axis was measured by nothing.
    # test_fixed_batch_exercises_every_clip_regime pins this property so it
    # cannot silently decay again.
    old = tuple(tuple(value + rng.uniform(-0.30, 0.30) for value in row) for row in current)
    mask = (
        (1, 1, 1, 1, 1, 1),
        (1, 1, 1, 1, 0, 0),
        (1, 1, 1, 1, 1, 0),
        (1, 1, 0, 0, 0, 0),
    )
    return prompt_ids, rewards, current, old, mask


def _fixed_reference(current: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """A fixed reference plane offset from ``current``, for an active KL.

    The 0.5 half-width is load-bearing. k3(x) = expm1(x) - x is symmetric to
    SECOND order, so at a small offset k3(x) and k3(-x) differ only by
    x**3/3. At the original 0.02 that gap was ~1e-9 -- a thousand times
    under the 1e-6 tolerance -- and a kernel computing the KL in the WRONG
    DIRECTION (current - reference instead of reference - current) agreed
    with the oracle anyway. A mutation reversing the direction survived the
    whole suite. At 0.5 the third-order term is large enough that the
    direction is a measured property rather than an assumed one.
    """
    rng = random.Random(1)
    return tuple(tuple(value + rng.uniform(-0.5, 0.5) for value in row) for row in current)


def _oracle_and_tensor_losses(objective: Any) -> tuple[float, torch.Tensor]:
    """Price one batch through the oracle AND the tensor kernel identically.

    The oracle runs its own ``forward_fn`` pipeline over the full batch;
    the tensor kernel receives exactly the estimator's USED rows, with the
    estimator's own per-token weight rows as ``advantages``, so the only
    quantity under test is the kernel's arithmetic.

    The reference plane is supplied to BOTH planes exactly when the objective
    declares a non-zero ``kl_weight``. It is not supplied unconditionally:
    a reference-free objective that silently accepted one would hide the
    refusal that is supposed to fire when a k3 term has no reference.
    """
    prompt_ids, rewards, current, old, mask = _fixed_batch()
    columns: dict[str, Any] = {
        "prompt_ids": prompt_ids,
        "rewards": rewards,
        "loss_mask": mask,
        "old_logprobs": old,
    }
    reference: tuple[tuple[float, ...], ...] | None = None
    # getattr, not attribute access: DAPO omits kl_weight STRUCTURALLY rather
    # than declaring it zero, so the attribute is absent, not falsy. This is
    # the same read the kernel performs, and reading it any other way would
    # make the test disagree with the thing it measures.
    if float(getattr(objective, "kl_weight", 0.0)) != 0.0:
        reference = _fixed_reference(current)
        columns["reference_logprobs"] = reference
    batch = ExperienceBatch(columns=columns, required=tuple(columns))

    def forward_fn(_batch: ExperienceBatch) -> Sequence[Sequence[float]]:
        return current

    oracle_output = objective(forward_fn, batch)

    advantage = objective.advantage_fn.compute(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=mask,
    )
    used = advantage.rows
    tensor_loss = TensorPolicyLoss(objective)(
        current_logprobs=torch.tensor([current[row] for row in used], requires_grad=True),
        old_logprobs=torch.tensor([old[row] for row in used]),
        advantages=torch.tensor([list(weights) for weights in advantage.weights]),
        mask=torch.tensor([mask[row] for row in used], dtype=torch.float64),
        reference_logprobs=(
            None if reference is None else torch.tensor([reference[row] for row in used])
        ),
    )
    return oracle_output.loss, tensor_loss


def test_grpo_tensor_loss_matches_oracle() -> None:
    oracle_loss, tensor_loss = _oracle_and_tensor_losses(GRPOPolicyLoss(group_size=2))
    assert abs(oracle_loss - float(tensor_loss)) <= 1e-6


def test_gspo_tensor_loss_matches_oracle() -> None:
    oracle_loss, tensor_loss = _oracle_and_tensor_losses(GSPOLoss(group_size=2))
    assert abs(oracle_loss - float(tensor_loss)) <= 1e-6


def test_dr_grpo_tensor_loss_matches_oracle() -> None:
    oracle_loss, tensor_loss = _oracle_and_tensor_losses(DrGRPOLoss(group_size=2))
    assert abs(oracle_loss - float(tensor_loss)) <= 1e-6


def test_dapo_tensor_loss_matches_oracle() -> None:
    oracle_loss, tensor_loss = _oracle_and_tensor_losses(DAPOLoss(group_size=2))
    assert abs(oracle_loss - float(tensor_loss)) <= 1e-6


def test_gspo_with_active_kl_matches_oracle() -> None:
    # The one leg that exercises the k3 branch on a SEQUENCE-scope ratio:
    # GSPO declares kl_weight=0.0 by default, so without this override the
    # kernel's KL arithmetic is reachable only through GRPO's token scope.
    oracle_loss, tensor_loss = _oracle_and_tensor_losses(GSPOLoss(group_size=2, kl_weight=0.01))
    assert abs(oracle_loss - float(tensor_loss)) <= 1e-6


def test_row_vector_advantages_match_per_token_form() -> None:
    # GroupNormalisedAdvantage broadcasts one constant weight per row, so
    # the (rows,) and (rows, tokens) advantage forms must agree exactly.
    objective = GRPOPolicyLoss(group_size=2)
    prompt_ids, rewards, current, old, mask = _fixed_batch()
    reference = _fixed_reference(current)
    advantage = objective.advantage_fn.compute(prompt_ids=prompt_ids, rewards=rewards, mask=mask)
    used = advantage.rows
    per_token = TensorPolicyLoss(objective)(
        current_logprobs=torch.tensor([current[row] for row in used], requires_grad=True),
        old_logprobs=torch.tensor([old[row] for row in used]),
        advantages=torch.tensor([list(weights) for weights in advantage.weights]),
        mask=torch.tensor([mask[row] for row in used], dtype=torch.float64),
        reference_logprobs=torch.tensor([reference[row] for row in used]),
    )
    per_row = TensorPolicyLoss(objective)(
        current_logprobs=torch.tensor([current[row] for row in used], requires_grad=True),
        old_logprobs=torch.tensor([old[row] for row in used]),
        advantages=torch.tensor([weights[0] for weights in advantage.weights]),
        mask=torch.tensor([mask[row] for row in used], dtype=torch.float64),
        reference_logprobs=torch.tensor([reference[row] for row in used]),
    )
    assert abs(float(per_token) - float(per_row)) <= 1e-9


def test_gradient_flows_to_current_logprobs() -> None:
    prompt_ids, rewards, current, old, mask = _fixed_batch()
    objective = GRPOPolicyLoss(group_size=2)
    reference = _fixed_reference(current)
    advantage = objective.advantage_fn.compute(prompt_ids=prompt_ids, rewards=rewards, mask=mask)
    used = advantage.rows
    current_tensor = torch.tensor([current[row] for row in used], requires_grad=True)
    loss = TensorPolicyLoss(objective)(
        current_logprobs=current_tensor,
        old_logprobs=torch.tensor([old[row] for row in used]),
        advantages=torch.tensor([list(weights) for weights in advantage.weights]),
        mask=torch.tensor([mask[row] for row in used], dtype=torch.float64),
        reference_logprobs=torch.tensor([reference[row] for row in used]),
    )
    assert loss.dim() == 0
    assert loss.requires_grad
    loss.backward()
    assert current_tensor.grad is not None
    assert bool((current_tensor.grad != 0.0).any())


def test_fully_masked_batch_refuses() -> None:
    prompt_ids, rewards, current, old, _mask = _fixed_batch()
    objective = GRPOPolicyLoss(group_size=2)
    advantage = objective.advantage_fn.compute(prompt_ids=prompt_ids, rewards=rewards, mask=_mask)
    used = advantage.rows
    with pytest.raises(BatchRefusal, match=r"0 of \d+ mask entries are supervised"):
        TensorPolicyLoss(objective)(
            current_logprobs=torch.tensor([current[row] for row in used], requires_grad=True),
            old_logprobs=torch.tensor([old[row] for row in used]),
            advantages=torch.tensor([list(weights) for weights in advantage.weights]),
            mask=torch.zeros((len(used), _TOKENS), dtype=torch.float64),
        )


def test_shape_mismatch_refuses_naming_both_sides() -> None:
    objective = GRPOPolicyLoss(group_size=2)
    current = torch.zeros((4, 6), requires_grad=True)
    old = torch.zeros((4, 3))
    with pytest.raises(BatchRefusal) as excinfo:
        TensorPolicyLoss(objective)(
            current_logprobs=current,
            old_logprobs=old,
            advantages=torch.zeros(4),
            mask=torch.ones((4, 6)),
        )
    message = str(excinfo.value)
    assert "(4, 6)" in message
    assert "(4, 3)" in message


def test_detached_current_logprobs_refuse() -> None:
    objective = GRPOPolicyLoss(group_size=2)
    with pytest.raises(BatchRefusal, match="requires_grad is False"):
        TensorPolicyLoss(objective)(
            current_logprobs=torch.zeros((4, 6)),
            old_logprobs=torch.zeros((4, 6)),
            advantages=torch.zeros(4),
            mask=torch.ones((4, 6)),
        )


def test_fixed_batch_exercises_every_clip_regime() -> None:
    """Pin the batch's ADEQUACY, not just the kernel's arithmetic.

    Every equivalence test above is only as strong as the batch it prices.
    If the ratios drift back inside all the clip bounds, the equivalences
    keep passing while measuring a kernel with no clipping at all. This
    test fails the moment that happens, and names which regime went dark.
    """
    _prompt_ids, _rewards, current, old, _mask = _fixed_batch()
    ratios = [
        math.exp(c - o)
        for current_row, old_row in zip(current, old, strict=True)
        for c, o in zip(current_row, old_row, strict=True)
    ]
    below = sum(1 for r in ratios if r < 0.8)
    above_grpo = sum(1 for r in ratios if r > 1.2)
    distinguishing = sum(1 for r in ratios if 1.2 < r <= 1.28)
    assert below, f"0 of {len(ratios)} ratios fall below the 0.8 lower clip"
    assert above_grpo, f"0 of {len(ratios)} ratios rise above GRPO's 1.2 upper clip"
    # The band where GRPO clips and DAPO does not. Without it DAPO's
    # asymmetric bound is arithmetically indistinguishable from GRPO's.
    assert distinguishing, (
        f"0 of {len(ratios)} ratios land in (1.2, 1.28], so DAPO's "
        f"asymmetric upper clip is not distinguished from GRPO's"
    )


def test_mask_is_load_bearing_for_gradient() -> None:
    """Masked positions must receive exactly zero gradient.

    Every equivalence leg feeds the estimator's own weight rows, which
    already carry 0.0 at masked positions -- so a kernel that dropped its
    ``* mask_f`` entirely would agree with the oracle on all of them. This
    leg supplies UNIFORM advantages, where the mask is the only thing that
    can zero a masked position, and asserts on the gradient rather than
    the loss so that a masked token contributing zero by cancellation
    cannot be mistaken for one excluded by the mask.
    """
    _prompt_ids, _rewards, current, old, mask = _fixed_batch()
    objective = GRPOPolicyLoss(group_size=2)
    current_tensor = torch.tensor(current, requires_grad=True)
    loss = TensorPolicyLoss(objective)(
        current_logprobs=current_tensor,
        old_logprobs=torch.tensor(old),
        advantages=torch.ones((_ROWS, _TOKENS), dtype=torch.float32),
        mask=torch.tensor(mask, dtype=torch.float32),
        reference_logprobs=torch.tensor(_fixed_reference(current)),
    )
    loss.backward()
    grad = current_tensor.grad
    assert grad is not None
    for row, mask_row in enumerate(mask):
        for position, supervised in enumerate(mask_row):
            observed = float(grad[row, position])
            if supervised:
                assert observed != 0.0, f"supervised ({row}, {position}) took no gradient"
            else:
                assert observed == 0.0, (
                    f"masked ({row}, {position}) took gradient {observed!r}; "
                    f"the supervision mask is not gating the loss"
                )


def test_sequence_scope_collapses_advantages_before_clipping() -> None:
    """A sequence-scope ratio must clip ONE pair per response.

    Every equivalence leg above feeds advantages that are constant within a
    row, and on those the collapse is an exact identity -- so none of them
    can tell whether the kernel clips per response or per token. The two
    forms diverge only when the per-token weights vary, which is what a
    token-level advantage estimator would supply. GSPO's definition clips
    the sequence ratio against the SEQUENCE advantage, so this leg prices a
    varying-advantage row against that definition computed independently in
    plain Python, rather than against the shipped oracle -- the oracle
    cannot be driven to this input through its own estimator.
    """
    objective = GSPOLoss(group_size=2)
    clip_low, clip_high = objective.clip_bounds
    current = ((-1.0, -2.0, -0.5, -1.5),)
    old = ((-1.2, -1.7, -0.9, -1.4),)
    mask = ((1, 1, 1, 0),)
    # Signs differ inside the row: this is exactly the case where
    # mean_t min(r*a_t, c*a_t) != min(r*abar, c*abar).
    advantages = ((2.0, -3.0, 1.0, 0.0),)

    supervised = float(sum(mask[0]))
    sequence_log_ratio = (
        sum((c - o) * m for c, o, m in zip(current[0], old[0], mask[0], strict=True)) / supervised
    )
    ratio = math.exp(sequence_log_ratio)
    clipped = min(max(ratio, clip_low), clip_high)
    row_advantage = sum(a * m for a, m in zip(advantages[0], mask[0], strict=True)) / supervised
    # sequence_mean over one row, then the objective's own weight and sign.
    expected = -float(objective.weight) * min(ratio * row_advantage, clipped * row_advantage)

    observed = TensorPolicyLoss(objective)(
        current_logprobs=torch.tensor(current, dtype=torch.float64, requires_grad=True),
        old_logprobs=torch.tensor(old, dtype=torch.float64),
        advantages=torch.tensor(advantages, dtype=torch.float64),
        mask=torch.tensor(mask, dtype=torch.float64),
    )
    assert abs(expected - float(observed)) <= 1e-9


def test_row_with_no_supervised_token_refuses() -> None:
    objective = GRPOPolicyLoss(group_size=2)
    mask = torch.ones((3, 4), dtype=torch.float64)
    mask[1] = 0.0
    with pytest.raises(BatchRefusal, match=r"1 of 3 rows carry 0 supervised tokens"):
        TensorPolicyLoss(objective)(
            current_logprobs=torch.zeros((3, 4), requires_grad=True),
            old_logprobs=torch.zeros((3, 4)),
            advantages=torch.ones(3),
            mask=mask,
        )


def test_overflowing_ratio_refuses_rather_than_returning_nan() -> None:
    # Inputs are finite, so the input guards pass; exp() of a log-ratio of
    # 800 overflows inside the kernel. Without the scalar finiteness check
    # this returns a non-finite loss that poisons every parameter on the
    # first backward.
    #
    # The advantage must be NEGATIVE for the overflow to escape. With a
    # positive advantage the PPO min absorbs it -- min(inf, 1.2) is 1.2, so
    # the clip itself bounds the term. It is only on the negative branch,
    # where the unclipped product is -inf and the min selects it, that the
    # infinity reaches the reduction. Dr.GRPO is used because it is
    # reference-free: GRPO refuses a zero kl_weight at construction.
    objective = DrGRPOLoss(group_size=2)
    current = torch.full((2, 3), 800.0, requires_grad=True)
    with pytest.raises(BatchRefusal, match="non-finite loss"):
        TensorPolicyLoss(objective)(
            current_logprobs=current,
            old_logprobs=torch.zeros((2, 3)),
            advantages=torch.full((2,), -1.0),
            mask=torch.ones((2, 3)),
        )


class _UndeclaredScope:
    """A stub objective declaring an axis value the kernel does not know."""

    ratio_scope = "chunk"
    reduction = "token_mean"
    clip_bounds = (0.8, 1.2)


class _UndeclaredReduction:
    ratio_scope = "token"
    reduction = "harmonic"
    clip_bounds = (0.8, 1.2)


class _NonPositiveConstantLength:
    ratio_scope = "token"
    reduction = "constant"
    clip_bounds = (0.8, 1.2)
    constant_length = 0


@pytest.mark.parametrize(
    ("objective", "pattern"),
    [
        (_UndeclaredScope(), r"ratio_scope='chunk'"),
        (_UndeclaredReduction(), r"reduction='harmonic'"),
        (_NonPositiveConstantLength(), r"constant_length=0"),
    ],
)
def test_undeclared_axis_values_refuse(objective: Any, pattern: str) -> None:
    with pytest.raises(BatchRefusal, match=pattern):
        TensorPolicyLoss(objective)(
            current_logprobs=torch.zeros((2, 3), requires_grad=True),
            old_logprobs=torch.zeros((2, 3)),
            advantages=torch.ones(2),
            mask=torch.ones((2, 3)),
        )


def test_active_kl_without_reference_refuses() -> None:
    objective = GRPOPolicyLoss(group_size=2)
    with pytest.raises(BatchRefusal, match="required reference_logprobs"):
        TensorPolicyLoss(objective)(
            current_logprobs=torch.zeros((2, 3), requires_grad=True),
            old_logprobs=torch.zeros((2, 3)),
            advantages=torch.ones(2),
            mask=torch.ones((2, 3)),
        )


def test_non_finite_input_refuses() -> None:
    objective = GRPOPolicyLoss(group_size=2)
    old = torch.zeros((2, 3))
    old[0, 1] = float("nan")
    with pytest.raises(BatchRefusal, match=r"1 of 6 entries in old_logprobs"):
        TensorPolicyLoss(objective)(
            current_logprobs=torch.zeros((2, 3), requires_grad=True),
            old_logprobs=old,
            advantages=torch.ones(2),
            mask=torch.ones((2, 3)),
        )
