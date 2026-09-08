"""Tests for the stage-2 DPO loss."""

from __future__ import annotations

import math
from typing import Any

import pytest

from foundationscale.gates.objective_gates import MetricObservation
from foundationscale.rl import (
    BatchRefusal,
    DPOLoss,
    ExperienceBatch,
    ForwardFn,
    LossConfigRefusal,
    SupervisionRefusal,
)

# A healthy pair: the policy scores the chosen completion well above the
# rejected one, and both reference scores sit at 0.0 so every margin is
# positive. Masks supervise every token.
_GOOD_CHOSEN = ((-0.1, -0.2), (-0.15, -0.25))
_GOOD_REJECTED = ((-0.5, -0.6), (-0.55, -0.65))
_GOOD_MASKS = ((1, 1), (1, 1))
_ZERO_REFS = (0.0, 0.0)


def _batch(
    chosen_masks: Any = _GOOD_MASKS,
    rejected_masks: Any = _GOOD_MASKS,
    ref_chosen: Any = _ZERO_REFS,
    ref_rejected: Any = _ZERO_REFS,
) -> ExperienceBatch:
    return ExperienceBatch(
        columns={
            "chosen_loss_mask": chosen_masks,
            "rejected_loss_mask": rejected_masks,
            "reference_chosen_logprob": ref_chosen,
            "reference_rejected_logprob": ref_rejected,
        }
    )


def _good_batch() -> ExperienceBatch:
    return _batch()


def _forward(rows: Any) -> ForwardFn:
    materialised = list(rows)
    return lambda batch: list(materialised)


def _good_forward() -> ForwardFn:
    return _forward(zip(_GOOD_CHOSEN, _GOOD_REJECTED, strict=True))


def test_dpo_loss_is_reexported_from_the_public_path() -> None:
    import foundationscale.rl as rl_public
    import foundationscale.rl.losses as losses_module

    assert rl_public.DPOLoss is losses_module.DPOLoss


def test_healthy_batch_prefers_chosen() -> None:
    out = DPOLoss()(_good_forward(), _good_batch())
    assert math.isfinite(out.loss)
    assert out.loss > 0.0
    assert [c.name for c in out.components] == ["preference_loss"]
    assert out.components[0].observed is True
    assert out.components[0].contribution == out.loss
    assert len(out.metrics) == 1
    assert out.metrics[0].name == "accuracy"
    assert out.metrics[0].value == 1.0


def test_zero_margin_is_log_two_and_counts_as_incorrect() -> None:
    # Policy scores equal the reference scores on both sides, so the margin is
    # exactly 0.0 for every pair: -log sigmoid(0) == log 2, and the tie counts
    # as NOT a correct ranking.
    rows = [((-0.3, -0.4), (-0.6, -0.1)), ((-0.3, -0.4), (-0.6, -0.1))]
    batch = _batch(ref_chosen=(-0.7, -0.7), ref_rejected=(-0.7, -0.7))
    out = DPOLoss()(_forward(rows), batch)
    assert out.loss == pytest.approx(math.log(2))
    assert out.metrics[0].value == 0.0


def test_phase2_anomaly_pinned_zero_accuracy_is_the_declared_degenerate_reading() -> None:
    # The policy ranks the REJECTED completion above the chosen one on every
    # pair: accuracy pins at 0.0, which bounds alone cannot refuse -- it is
    # inside accuracy's natural range -- so it must be the declared degenerate.
    rejected_favoured_chosen = ((-0.9, -0.9), (-0.95, -0.95))
    rejected_favoured_rejected = ((-0.1, -0.1), (-0.05, -0.05))
    rows = zip(rejected_favoured_chosen, rejected_favoured_rejected, strict=True)
    out = DPOLoss()(_forward(rows), _good_batch())
    metric = out.metrics[0]
    assert metric == MetricObservation(name="accuracy", value=0.0)
    expectation = DPOLoss().declaration().metrics[0]
    assert expectation.degenerate == (0.0,)
    assert metric.value in expectation.degenerate


def test_declaration_omits_sft_when_sft_weight_is_zero() -> None:
    declaration = DPOLoss(sft_weight=0.0).declaration()
    assert declaration.components == ("preference_loss",)
    assert "sft_loss" not in declaration.components
    expectation = declaration.metrics[0]
    assert expectation.name == "accuracy"
    assert expectation.low == 0.0
    assert expectation.high == 1.0
    assert expectation.degenerate == (0.0,)


def test_declaration_includes_sft_when_sft_weight_is_nonzero() -> None:
    declaration = DPOLoss(sft_weight=0.5).declaration()
    assert declaration.components == ("preference_loss", "sft_loss")


@pytest.mark.parametrize("sft_weight", (0.0, 0.5))
def test_declared_and_observed_components_cannot_diverge(sft_weight: float) -> None:
    # Condition (a) control: declaration and computation read the ONE field
    # sft_weight, so the observed component set equals the declared set under
    # both settings. There is no computed-but-undeclared state to be blind to.
    loss_fn = DPOLoss(sft_weight=sft_weight)
    out = loss_fn(_good_forward(), _good_batch())
    declared = set(loss_fn.declaration().components)
    observed = {component.name for component in out.components}
    assert observed == declared


def test_extreme_negative_margin_stays_finite() -> None:
    # margin == 0.1 * ((-4000 - 0) - (4000 - 0)) == -800; the naive
    # -log(1 / (1 + exp(-margin))) overflows exactly here, which is the regime
    # a diverging run enters. The stable softplus branch keeps the loss finite.
    rows = [((-4000.0,), (4000.0,))]
    batch = _batch(((1,),), ((1,),), (0.0,), (0.0,))
    out = DPOLoss()(_forward(rows), batch)
    assert math.isfinite(out.loss)
    assert out.loss == pytest.approx(800.0)


def test_sequence_scores_are_sums_not_means() -> None:
    # Two pairs identical except the second has its chosen completion duplicated
    # to twice the length. A length-normalised mean would give both the same
    # margin; a sum does not. This pins that DPO compares sequence likelihoods.
    loss_fn = DPOLoss()
    short_rows = [((-0.5,), (-1.0,))]
    short_batch = _batch(((1,),), ((1,),), (0.0,), (0.0,))
    long_rows = [((-0.5, -0.5), (-1.0,))]
    long_batch = _batch(((1, 1),), ((1,),), (0.0,), (0.0,))
    short_loss = loss_fn(_forward(short_rows), short_batch).loss
    long_loss = loss_fn(_forward(long_rows), long_batch).loss
    # Short pair: margin == 0.1 * (0.5) == 0.05. Long pair: the duplicated
    # chosen side sums to -1.0, so margin == 0.1 * ((-1.0) - (-1.0)) == 0.0.
    assert short_loss == pytest.approx(math.log1p(math.exp(-0.05)))
    assert long_loss == pytest.approx(math.log(2))
    assert short_loss != pytest.approx(long_loss)


def test_missing_reference_column_refused() -> None:
    batch = ExperienceBatch(
        columns={
            "chosen_loss_mask": _GOOD_MASKS,
            "rejected_loss_mask": _GOOD_MASKS,
            "reference_chosen_logprob": _ZERO_REFS,
        }
    )
    with pytest.raises(BatchRefusal) as exc_info:
        DPOLoss()(_good_forward(), batch)
    assert "reference_rejected_logprob" in str(exc_info.value)


def test_empty_batch_refused() -> None:
    batch = _batch((), (), (), ())
    with pytest.raises(BatchRefusal) as exc_info:
        DPOLoss()(_forward([]), batch)
    assert "0" in str(exc_info.value)


def test_wrong_forward_row_count_refused() -> None:
    rows = [((-0.5,), (-0.5,))]  # one row for a two-row batch
    with pytest.raises(BatchRefusal) as exc_info:
        DPOLoss()(_forward(rows), _good_batch())
    message = str(exc_info.value)
    assert "1" in message
    assert "2" in message


def test_row_that_is_not_a_pair_refused() -> None:
    # Row 0 must be WELL FORMED, or its own mask-length refusal fires first and
    # this test passes without ever reaching the arity check it exists for.
    rows = [
        (_GOOD_CHOSEN[0], _GOOD_REJECTED[0]),
        ((-0.1, -0.2), (-0.5, -0.6), (-0.9, -1.0)),
    ]
    with pytest.raises(BatchRefusal) as exc_info:
        DPOLoss()(_forward(rows), _good_batch())
    message = str(exc_info.value)
    assert "1" in message  # the row index
    assert "3" in message  # the length it got


def test_mask_length_disagreeing_with_token_count_refused() -> None:
    batch = _batch(chosen_masks=((1,), (1, 1)))
    with pytest.raises(BatchRefusal) as exc_info:
        DPOLoss()(_good_forward(), batch)
    message = str(exc_info.value)
    assert "0" in message  # the row index
    assert "chosen" in message
    assert "2" in message
    assert "1" in message


def test_fractional_mask_entry_refused() -> None:
    batch = _batch(chosen_masks=((0.5, 1), (1, 1)))
    with pytest.raises(BatchRefusal) as exc_info:
        DPOLoss()(_good_forward(), batch)
    message = str(exc_info.value)
    assert "0.5" in message
    assert "0 or 1" in message


def test_zero_supervision_side_refused() -> None:
    batch = _batch(rejected_masks=((0, 0), (1, 1)))
    with pytest.raises(SupervisionRefusal) as exc_info:
        DPOLoss()(_good_forward(), batch)
    message = str(exc_info.value)
    assert "rejected" in message
    assert "0" in message  # the row index and the supervised-token count


def test_non_finite_reference_value_refused() -> None:
    batch = _batch(ref_chosen=(0.0, float("nan")))
    with pytest.raises(BatchRefusal) as exc_info:
        DPOLoss()(_good_forward(), batch)
    message = str(exc_info.value)
    assert "reference_chosen_logprob" in message
    assert "1" in message  # the row index
    assert "nan" in message


def test_beta_zero_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        DPOLoss(beta=0.0)
    assert "0.0" in str(exc_info.value)


def test_beta_negative_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        DPOLoss(beta=-1.0)
    assert "-1.0" in str(exc_info.value)


def test_weight_zero_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        DPOLoss(weight=0.0)
    assert "0.0" in str(exc_info.value)


def test_sft_weight_negative_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        DPOLoss(sft_weight=-1.0)
    assert "-1.0" in str(exc_info.value)


def test_component_names_must_not_collide() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        DPOLoss(component_name="shared", sft_component_name="shared")
    assert "shared" in str(exc_info.value)


@pytest.mark.parametrize(
    "field",
    [
        "chosen_mask_column",
        "rejected_mask_column",
        "reference_chosen_column",
        "reference_rejected_column",
        "component_name",
        "sft_component_name",
        "metric_name",
    ],
)
@pytest.mark.parametrize("bad", ["", 7])
def test_every_name_field_refuses_a_non_name(field: str, bad: object) -> None:
    # Parametrised over the WHOLE field list, not a sample: a name field added
    # later without a guard is the case this catches, and one representative
    # field would not catch it.
    with pytest.raises(LossConfigRefusal) as exc_info:
        DPOLoss(**{field: bad})  # type: ignore[arg-type]
    assert field in str(exc_info.value)


def test_row_that_is_not_a_sized_sequence_refused() -> None:
    with pytest.raises(BatchRefusal) as exc_info:
        DPOLoss()(_forward([7, (_GOOD_CHOSEN[1], _GOOD_REJECTED[1])]), _good_batch())
    assert "row 0" in str(exc_info.value)


def test_unconvertible_mask_entry_refused() -> None:
    batch = _batch(chosen_masks=((1, object()), (1, 1)))
    with pytest.raises(BatchRefusal) as exc_info:
        DPOLoss()(_good_forward(), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "position 1" in message


def test_unconvertible_log_probability_refused() -> None:
    rows = [((-0.1, object()), _GOOD_REJECTED[0]), (_GOOD_CHOSEN[1], _GOOD_REJECTED[1])]
    with pytest.raises(BatchRefusal) as exc_info:
        DPOLoss()(_forward(rows), _good_batch())
    message = str(exc_info.value)
    assert "row 0" in message
    assert "position 1" in message


def test_unconvertible_reference_score_refused() -> None:
    batch = _batch(ref_chosen=(object(), 0.0))
    with pytest.raises(BatchRefusal) as exc_info:
        DPOLoss()(_good_forward(), batch)
    message = str(exc_info.value)
    assert "reference_chosen_logprob" in message
    assert "row 0" in message
