# SPDX-License-Identifier: Apache-2.0
"""Tensor/float parity tests for the six preference objectives.

Every preference objective exists in two statements: the float oracle in
``losses.py`` / ``preference_objectives.py`` (the audited statement of the
mathematics) and the differentiable kernel ``TensorPreferenceLoss`` in
``preference_torch.py`` (which must evaluate the SAME mathematics). This
module feeds both planes the same synthetic per-token log-probabilities and
masks and refuses to let them disagree.

The float oracles are constructed with the six factories' default knobs
(``dpo_algorithm()`` builds ``DPOLoss(beta=0.1, weight=1.0, sft_weight=0.0)``
and so on); they are built from the classes directly rather than extracted
from a ``PreferenceAlgorithm`` because the algorithm exposes no public
objective accessor, and a test that reaches into ``_objective`` pins a
private name.

WHAT IS CLAIMED: for equivalent finite inputs, each kernel's scalar agrees
with its oracle's ``LossOutput.loss`` within 1e-6 (both in float64); the
gradient is finite, nonzero, and lands only on ``policy_logprobs``; and the
refusal cases -- non-finite input, a zero-supervision side, a missing
reference, a reference handed to a reference-free objective, KTO's missing
reference point -- are refused by the plane that can observe them, with the
two planes agreeing wherever both can.

WHAT IS NOT CLAIMED: that any model forward, reference model, or data
producer exists here -- the synthetic tensors are fixed-seed readings, not
measurements of a trained model -- or that parity holds for objectives
outside the six the kernel's recipe table admits.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from foundationscale.rl.interfaces import (  # noqa: E402
    BatchRefusal,
    LossConfigRefusal,
    SupervisionRefusal,
)
from foundationscale.rl.losses import DPOLoss  # noqa: E402
from foundationscale.rl.preference_objectives import (  # noqa: E402
    CPOLoss,
    IPOLoss,
    KTOLoss,
    ORPOLoss,
    SimPOLoss,
)
from foundationscale.rl.preference_torch import (  # noqa: E402
    TensorPreferenceLoss,
    is_paired,
    needs_reference,
    preference_metrics,
)

# Fixed shapes and seeds: parity is an arithmetic claim, so the run must not
# depend on sampling luck. Lengths vary within a batch through the MASKS --
# every length is >= 1 because a zero-supervision side is a refused input,
# not a testable value. Policy entries are bounded away from 0.0 below so
# ORPO's float64 odds wall (a length-normalised average of exactly 0.0) is
# never approached, and the sign stays negative as real log-probabilities do.
PAIR_COUNT = 4
TOKEN_COUNT = 7
CHOSEN_LENGTHS = (7, 1, 5, 2)
REJECTED_LENGTHS = (3, 7, 4, 6)

KTO_ROW_COUNT = 5
KTO_LENGTHS = (7, 2, 5, 1, 6)
KTO_DESIRABLE = (True, False, True, False, True)
KTO_REFERENCE_POINT = 0.35

_PAIRED_OBJECTIVES: tuple[tuple[str, Any], ...] = (
    ("dpo", DPOLoss()),
    ("ipo", IPOLoss()),
    ("simpo", SimPOLoss()),
    ("orpo", ORPOLoss()),
    ("cpo", CPOLoss()),
)


class _Batch:
    """A minimal ExperienceBatch-shaped carrier for the float oracles.

    The six oracles read exactly three things from a batch: ``columns``,
    ``column(name)``, and ``len(batch)``. The real ``ExperienceBatch`` adds
    schema validation this module deliberately does not need, because the
    refusal cases under test must reach the ORACLE's own checks rather than
    be refused one layer earlier by a constructor.
    """

    def __init__(self, columns: dict[str, list[Any]], row_count: int) -> None:
        self._columns = dict(columns)
        self._row_count = row_count

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._columns)

    def column(self, name: str) -> list[Any]:
        return self._columns[name]

    def __len__(self) -> int:
        return self._row_count


def _prefix_mask(lengths: tuple[int, ...], width: int) -> torch.Tensor:
    mask = torch.zeros(len(lengths), width, dtype=torch.float64)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1.0
    return mask


def _interleave(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Interleave as [chosen_0, rejected_0, chosen_1, rejected_1, ...]."""
    stacked = torch.stack((left, right), dim=1)
    return stacked.reshape(2 * int(left.shape[0]), int(left.shape[1]))


def _paired_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interleaved policy/reference readings and masks, identical every call."""
    torch.manual_seed(5821)
    policy = -(0.05 + 3.0 * torch.rand(2 * PAIR_COUNT, TOKEN_COUNT, dtype=torch.float64))
    reference = -(0.05 + 3.0 * torch.rand(2 * PAIR_COUNT, TOKEN_COUNT, dtype=torch.float64))
    mask = _interleave(
        _prefix_mask(CHOSEN_LENGTHS, TOKEN_COUNT),
        _prefix_mask(REJECTED_LENGTHS, TOKEN_COUNT),
    )
    return policy, reference, mask


def _kto_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(9182)
    policy = -(0.05 + 3.0 * torch.rand(KTO_ROW_COUNT, TOKEN_COUNT, dtype=torch.float64))
    reference = -(0.05 + 3.0 * torch.rand(KTO_ROW_COUNT, TOKEN_COUNT, dtype=torch.float64))
    mask = _prefix_mask(KTO_LENGTHS, TOKEN_COUNT)
    desirable = torch.tensor(KTO_DESIRABLE, dtype=torch.bool)
    return policy, reference, mask, desirable


def _masked_sequence_sums(rows: list[list[float]], masks: list[list[float]]) -> list[float]:
    """The sequence-level sums the reference columns carry, one per batch row."""
    return [
        sum(value for value, keep in zip(row, mask_row, strict=True) if keep)
        for row, mask_row in zip(rows, masks, strict=True)
    ]


def _oracle_paired_loss(
    objective: Any,
    policy: torch.Tensor,
    mask: torch.Tensor,
    reference: torch.Tensor | None,
) -> float:
    """Price the interleaved tensors through the float oracle's own contract.

    The paired forward contract is one outer entry per batch row, each a
    2-element ``(chosen_token_logprobs, rejected_token_logprobs)`` sequence;
    the interleaved tensor order is de-interleaved back into that shape. The
    reference columns are per-row sequence-level SUMS, derived from the same
    reference tensor with the same masks -- matching how the anchored oracles
    define the column. When ``reference`` is None the columns are omitted
    entirely, which is how the missing-reference refusal is driven.
    """
    token_rows: list[list[float]] = policy.tolist()
    mask_rows: list[list[float]] = mask.tolist()
    chosen_tokens = token_rows[0::2]
    rejected_tokens = token_rows[1::2]
    chosen_masks = mask_rows[0::2]
    rejected_masks = mask_rows[1::2]
    columns: dict[str, list[Any]] = {
        objective.chosen_mask_column: chosen_masks,
        objective.rejected_mask_column: rejected_masks,
    }
    if not objective.reference_free and reference is not None:
        reference_rows: list[list[float]] = reference.tolist()
        columns[objective.reference_chosen_column] = _masked_sequence_sums(
            reference_rows[0::2], chosen_masks
        )
        columns[objective.reference_rejected_column] = _masked_sequence_sums(
            reference_rows[1::2], rejected_masks
        )

    def forward_fn(_batch: Any) -> list[tuple[list[float], list[float]]]:
        return list(zip(chosen_tokens, rejected_tokens, strict=True))

    output = objective(forward_fn, _Batch(columns, len(chosen_tokens)))
    return float(output.loss)


def _oracle_kto_loss(
    objective: KTOLoss,
    policy: torch.Tensor,
    mask: torch.Tensor,
    reference: torch.Tensor,
    desirable: torch.Tensor,
    *,
    include_reference_point: bool = True,
) -> float:
    """Price unpaired rows through KTO's contract: one completion per row."""
    token_rows: list[list[float]] = policy.tolist()
    mask_rows: list[list[float]] = mask.tolist()
    columns: dict[str, list[Any]] = {
        objective.mask_column: mask_rows,
        objective.reference_column: _masked_sequence_sums(reference.tolist(), mask_rows),
        objective.desirable_column: desirable.tolist(),
    }
    if include_reference_point:
        columns[objective.kl_reference_point_column] = [KTO_REFERENCE_POINT] * len(token_rows)

    def forward_fn(_batch: Any) -> list[list[float]]:
        return list(token_rows)

    output = objective(forward_fn, _Batch(columns, len(token_rows)))
    return float(output.loss)


@pytest.mark.parametrize(
    ("name", "objective"), _PAIRED_OBJECTIVES, ids=[n for n, _ in _PAIRED_OBJECTIVES]
)
def test_paired_kernel_matches_float_oracle(name: str, objective: Any) -> None:
    """Scalar parity for the five paired objectives on varied-length masks.

    WHAT IS CLAIMED: the kernel's returned scalar equals the oracle's
    ``loss`` within 1e-6, and the classification helpers agree with the
    objective's own ``reference_free`` declaration rather than a restated
    table.

    WHAT IS NOT CLAIMED: anything about the pairedness or reference-freeness
    of objectives these helpers were not handed.
    """
    assert is_paired(objective) is True, name
    assert needs_reference(objective) is (not objective.reference_free), name
    policy, reference, mask = _paired_tensors()
    reference_arg = None if objective.reference_free else reference
    kernel = TensorPreferenceLoss(objective)
    tensor_loss = kernel(
        policy_logprobs=policy.clone().requires_grad_(True),
        completion_mask=mask,
        reference_logprobs=reference_arg,
    )
    oracle_loss = _oracle_paired_loss(objective, policy, mask, reference)
    assert float(tensor_loss.item()) == pytest.approx(oracle_loss, rel=1e-6, abs=1e-6)


def test_kto_kernel_matches_float_oracle() -> None:
    """Scalar parity for the family's unpaired member.

    WHAT IS CLAIMED: the kernel's scalar equals the oracle's ``loss`` within
    1e-6 over rows of varied supervised length with both row families
    (desirable and undesirable) present, reading the SAME recorded reference
    point both planes are handed.

    WHAT IS NOT CLAIMED: that the reference point was estimated anywhere --
    it is a stated input on both sides, exactly as the objective demands.
    """
    objective = KTOLoss()
    assert is_paired(objective) is False
    assert needs_reference(objective) is True
    policy, reference, mask, desirable = _kto_tensors()
    kernel = TensorPreferenceLoss(objective)
    tensor_loss = kernel(
        policy_logprobs=policy.clone().requires_grad_(True),
        completion_mask=mask,
        reference_logprobs=reference,
        kl_reference_point=KTO_REFERENCE_POINT,
        desirable=desirable,
    )
    oracle_loss = _oracle_kto_loss(objective, policy, mask, reference, desirable)
    assert float(tensor_loss.item()) == pytest.approx(oracle_loss, rel=1e-6, abs=1e-6)


def test_gradient_is_finite_nonzero_and_lands_only_on_policy_logprobs() -> None:
    """The kernel is a differentiable statement, not a second oracle.

    WHAT IS CLAIMED for DPO: after ``backward()``, ``policy_logprobs.grad``
    is finite, is nonzero at every supervised position, is exactly zero at
    every masked-out position (masked readings carry no gradient, matching
    the oracle which never reads them), and the reference input accumulates
    NO gradient even when handed in as a grad-requiring tensor -- the kernel
    detaches it, as the frozen-reference seam demands.

    WHAT IS NOT CLAIMED: any particular gradient VALUE; sign and magnitude
    belong to the objective's formula, which the parity tests pin.
    """
    policy, reference, mask = _paired_tensors()
    policy_param = policy.clone().requires_grad_(True)
    reference_param = reference.clone().requires_grad_(True)
    kernel = TensorPreferenceLoss(DPOLoss())
    loss = kernel(
        policy_logprobs=policy_param,
        completion_mask=mask,
        reference_logprobs=reference_param,
    )
    loss.backward()
    gradient = policy_param.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    supervised = mask == 1.0
    assert bool(torch.all(gradient[supervised] != 0.0))
    assert bool(torch.all(gradient[~supervised] == 0.0))
    assert reference_param.grad is None, (
        "the reference tensor accumulated a gradient; the kernel must detach it, "
        "because a reference score that moves with the policy is not a reference"
    )


def test_kto_gradient_is_finite_and_nonzero_on_policy_logprobs() -> None:
    """The unpaired member's gradient reaches every supervised token.

    WHAT IS CLAIMED: with both row families present, the gradient is finite
    and nonzero at every supervised position -- a family whose labels
    silently stopped routing gradient would collapse this assertion.

    WHAT IS NOT CLAIMED: that desirable and undesirable rows carry gradients
    of the same sign; the two row families are weighted differently by
    construction.
    """
    policy, reference, mask, desirable = _kto_tensors()
    policy_param = policy.clone().requires_grad_(True)
    kernel = TensorPreferenceLoss(KTOLoss())
    loss = kernel(
        policy_logprobs=policy_param,
        completion_mask=mask,
        reference_logprobs=reference,
        kl_reference_point=KTO_REFERENCE_POINT,
        desirable=desirable,
    )
    loss.backward()
    gradient = policy_param.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert bool(torch.all(gradient[mask == 1.0] != 0.0))


def test_raising_chosen_logprobs_lowers_the_dpo_loss() -> None:
    """Sign probe: the loss must fall when the chosen side becomes likelier.

    Without this probe a kernel that had chosen and rejected transposed --
    the whole interleaving contract inverted -- would still pass scalar
    parity only if the oracle were wrong in the same way, and would pass
    shape checks unconditionally. The probe measures the SEAM, not the
    arithmetic: raising every supervised chosen token's log-probability
    (scaling the negative readings towards zero keeps them negative)
    increases the margin, and DPO's ``softplus(-margin)`` must decrease.

    WHAT IS NOT CLAIMED: any particular step size of the decrease.
    """
    policy, reference, mask = _paired_tensors()
    kernel = TensorPreferenceLoss(DPOLoss())
    base_loss = float(
        kernel(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=mask,
            reference_logprobs=reference,
        ).item()
    )
    raised = policy.clone()
    chosen_rows = torch.arange(0, 2 * PAIR_COUNT, 2)
    raised[chosen_rows] = raised[chosen_rows] * 0.25
    raised_loss = float(
        kernel(
            policy_logprobs=raised.requires_grad_(True),
            completion_mask=mask,
            reference_logprobs=reference,
        ).item()
    )
    assert raised_loss < base_loss, (
        f"raising the chosen side's log-probabilities moved the DPO loss from "
        f"{base_loss!r} to {raised_loss!r}; the interleaving seam is inverted "
        f"or the margin's sign is"
    )


def test_non_finite_policy_input_is_refused_on_both_planes() -> None:
    """Refusal parity: a NaN token reading is named, never propagated."""
    objective = DPOLoss()
    policy, reference, mask = _paired_tensors()
    poisoned = policy.clone()
    poisoned[0, 0] = float("nan")  # row 0 supervises from position 0
    with pytest.raises(BatchRefusal):
        TensorPreferenceLoss(objective)(
            policy_logprobs=poisoned.clone().requires_grad_(True),
            completion_mask=mask,
            reference_logprobs=reference,
        )
    with pytest.raises(BatchRefusal, match="not finite"):
        _oracle_paired_loss(objective, poisoned, mask, reference)


def test_zero_supervision_side_is_refused_on_both_planes() -> None:
    """Refusal parity: an empty completion is unmeasurable, on both planes."""
    objective = DPOLoss()
    policy, reference, mask = _paired_tensors()
    emptied = mask.clone()
    emptied[0] = 0.0  # interleaved row 0 is the chosen side of pair 0
    with pytest.raises(SupervisionRefusal, match="0 supervised tokens"):
        TensorPreferenceLoss(objective)(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=emptied,
            reference_logprobs=reference,
        )
    with pytest.raises(SupervisionRefusal, match="0 supervised tokens"):
        _oracle_paired_loss(objective, policy, emptied, reference)


def test_dpo_without_reference_input_is_refused_on_both_planes() -> None:
    """Refusal parity: the anchored margin cannot be formed unattributed.

    The kernel refuses the absent tensor; the oracle refuses the absent
    columns. Both refusals name the same fact -- a reference-anchored
    objective with no reference scores is unmeasurable -- through the
    observation channel each plane owns.
    """
    objective = DPOLoss()
    policy, reference, mask = _paired_tensors()
    with pytest.raises(BatchRefusal, match="reference_logprobs"):
        TensorPreferenceLoss(objective)(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=mask,
        )
    with pytest.raises(BatchRefusal, match="which are absent"):
        _oracle_paired_loss(objective, policy, mask, None)


def test_simpo_refuses_a_supplied_reference_tensor() -> None:
    """A reference-free objective must refuse reference readings, not read them.

    WHAT IS CLAIMED: handing ``reference_logprobs`` to SimPO is a refusal
    that names the declared ``reference_free`` seam, because readings wired
    to a recipe that never consumes them would sit silently unused.

    WHAT IS NOT CLAIMED: an oracle-plane refusal for this case -- the float
    SimPO's batch-schema contract demands only its two mask columns and has
    no channel through which a reference could be offered to it.
    """
    policy, reference, mask = _paired_tensors()
    with pytest.raises(BatchRefusal, match="reference_free"):
        TensorPreferenceLoss(SimPOLoss())(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=mask,
            reference_logprobs=reference,
        )


def test_kto_missing_reference_point_is_refused_on_both_planes() -> None:
    """Refusal parity: the KL reference point must be recorded, not estimated.

    The kernel refuses a ``None`` ``kl_reference_point``; the oracle refuses
    a batch whose recorded column is absent. Both refusals exist so a row's
    loss can never become a function of which other rows shared its batch.
    """
    objective = KTOLoss()
    policy, reference, mask, desirable = _kto_tensors()
    with pytest.raises(BatchRefusal, match="kl_reference_point"):
        TensorPreferenceLoss(objective)(
            policy_logprobs=policy.clone().requires_grad_(True),
            completion_mask=mask,
            reference_logprobs=reference,
            desirable=desirable,
        )
    with pytest.raises(BatchRefusal, match="KL reference-point"):
        _oracle_kto_loss(
            objective,
            policy,
            mask,
            reference,
            desirable,
            include_reference_point=False,
        )


def test_losses_dpo_refuses_a_nan_token_log_probability() -> None:
    """losses.py DPOLoss refuses a NaN reading with BatchRefusal, naming it.

    WHAT IS CLAIMED: a NaN at a SUPERVISED position of the forward output
    surfaces as ``BatchRefusal`` whose message says the reading is not
    finite -- never a NaN loss that no later check could attribute to its
    row, and never a silent clamp that would rewrite the row the model
    produced.

    WHAT IS NOT CLAIMED: that un-supervised positions are inspected -- the
    oracle reads only masked-in tokens, and ``_as_float`` is reached through
    ``_masked_side_score`` exactly at those.
    """
    objective = DPOLoss()
    policy, reference, mask = _paired_tensors()
    poisoned = policy.clone()
    poisoned[0, 0] = float("nan")
    with pytest.raises(BatchRefusal, match="which is not finite"):
        _oracle_paired_loss(objective, poisoned, mask, reference)


def test_ipo_length_normalised_kernel_matches_float_oracle() -> None:
    """Scalar parity for IPO's opt-in length-normalised margin.

    WHAT IS CLAIMED: with ``length_normalise=True`` the kernel's returned
    scalar equals the oracle's ``loss`` within 1e-6 on the varied-length
    fixture, where the chosen and rejected supervised-token counts differ
    within every pair -- so the normalisation is exercised, not vacuous.

    WHAT IS NOT CLAIMED: that the default recipe changed; the
    unnormalised parity case above still covers it bit-for-bit.
    """
    objective = IPOLoss(length_normalise=True)
    policy, reference, mask = _paired_tensors()
    tensor_loss = TensorPreferenceLoss(objective)(
        policy_logprobs=policy.clone().requires_grad_(True),
        completion_mask=mask,
        reference_logprobs=reference,
    )
    oracle_loss = _oracle_paired_loss(objective, policy, mask, reference)
    assert float(tensor_loss.item()) == pytest.approx(oracle_loss, rel=1e-6, abs=1e-6)


def test_ipo_length_normalise_changes_the_loss_when_side_lengths_differ() -> None:
    """The two margins are different objectives, not a flag's no-op.

    WHAT IS CLAIMED: on the varied-length fixture the summed (paper)
    margin and the normalised (TRL) margin produce DIFFERENT losses --
    within every fixture pair the chosen and rejected supervised counts
    differ, so dividing by them cannot be an identity transform, and a
    kernel or oracle that read the flag but never applied it would read
    equal here.

    WHAT IS NOT CLAIMED: which of the two losses is larger; that ordering
    is a property of the fixture's readings, not of the flag.
    """
    policy, reference, mask = _paired_tensors()
    summed = _oracle_paired_loss(IPOLoss(), policy, mask, reference)
    normalised = _oracle_paired_loss(IPOLoss(length_normalise=True), policy, mask, reference)
    assert summed != pytest.approx(normalised, rel=1e-9, abs=1e-9)


def test_ipo_normalised_margin_is_independent_of_length_for_uniform_tokens() -> None:
    """Equal per-token readings make the normalised h independent of length.

    With every supervised token at log-probability -0.5 (policy) and -0.2
    (reference), each side's anchored MEAN is exactly -0.3 regardless of
    how many tokens its mask selects, so the normalised margin is
    identically 0.0 and the loss is (1 / (2 * tau)) ** 2 on ANY mask
    layout. Two layouts with the side counts swapped must therefore
    agree, and the kernel's margin metric must report the h actually
    used -- 0.0 on both layouts.

    WHAT IS NOT CLAIMED: that the UNNORMALISED margin shares this
    property -- under masked sums, h would scale with the count
    difference, which is precisely the explosion the flag exists to avert.
    """
    objective = IPOLoss(length_normalise=True)
    width = 6
    policy = torch.full((4, width), -0.5, dtype=torch.float64)
    reference = torch.full((4, width), -0.2, dtype=torch.float64)
    mask_a = _interleave(_prefix_mask((6, 1), width), _prefix_mask((2, 5), width))
    mask_b = _interleave(_prefix_mask((1, 6), width), _prefix_mask((5, 2), width))
    target = 1.0 / (2.0 * float(objective.tau))
    loss_a = _oracle_paired_loss(objective, policy, mask_a, reference)
    loss_b = _oracle_paired_loss(objective, policy, mask_b, reference)
    assert loss_a == pytest.approx(target**2, rel=1e-12, abs=1e-12)
    assert loss_b == pytest.approx(loss_a, rel=1e-12, abs=1e-12)
    metrics_a = preference_metrics(objective, policy, mask_a, reference)
    metrics_b = preference_metrics(objective, policy, mask_b, reference)
    assert metrics_a["margin"] == pytest.approx(0.0, abs=1e-12)
    assert metrics_b["margin"] == pytest.approx(metrics_a["margin"], rel=1e-12, abs=1e-12)


def test_ipo_length_normalise_refuses_a_non_bool() -> None:
    """Config refusal: the flag must be a bool, not a truthy lookalike.

    WHAT IS CLAIMED: ``length_normalise=1`` refuses at construction with
    the offending value named -- every refusal names its reason --
    because a truthy non-bool would blur which margin the record says
    the run used.

    WHAT IS NOT CLAIMED: that any other objective accepts this knob; the
    flag is IPO's alone, and the trainer refuses it elsewhere through
    ``_unsupported_knobs``.
    """
    with pytest.raises(LossConfigRefusal, match="length_normalise"):
        IPOLoss(length_normalise=1)
