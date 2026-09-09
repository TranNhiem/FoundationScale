"""Tests for the reference-free preference objectives: ORPO, SimPO and CPO."""

from __future__ import annotations

import math
from typing import Any

import pytest

from foundationscale.gates.objective_gates import MetricObservation
from foundationscale.rl import (
    BatchRefusal,
    ExperienceBatch,
    ForwardFn,
    LossConfigRefusal,
    SupervisionRefusal,
)
from foundationscale.rl.preference_objectives import CPOLoss, ORPOLoss, SimPOLoss

ReferenceFreeLoss = ORPOLoss | SimPOLoss | CPOLoss

_TWO_ROW_MASKS = ((1, 1), (1, 1))
_GOOD_CHOSEN = ((-0.1, -0.2), (-0.15, -0.25))
_GOOD_REJECTED = ((-0.5, -0.6), (-0.55, -0.65))

PAIR_NAME_FIELDS = (
    "chosen_mask_column",
    "rejected_mask_column",
    "component_name",
    "sft_component_name",
    "accuracy_metric_name",
    "margin_metric_name",
)
SIMPO_NAME_FIELDS = (
    "chosen_mask_column",
    "rejected_mask_column",
    "component_name",
    "accuracy_metric_name",
    "margin_metric_name",
)


def _paired_batch(
    chosen_masks: Any = _TWO_ROW_MASKS,
    rejected_masks: Any = _TWO_ROW_MASKS,
    extra_columns: Any = None,
) -> ExperienceBatch:
    columns: dict[str, Any] = {
        "chosen_loss_mask": chosen_masks,
        "rejected_loss_mask": rejected_masks,
    }
    if extra_columns is not None:
        columns.update(extra_columns)
    return ExperienceBatch(columns=columns)


def _forward(rows: Any) -> ForwardFn:
    materialised = list(rows)
    return lambda _batch: list(materialised)


def _two_row_forward() -> ForwardFn:
    return _forward(zip(_GOOD_CHOSEN, _GOOD_REJECTED, strict=True))


# ---------------------------------------------------------------------------
# Configuration refusals -- ORPO.
# ---------------------------------------------------------------------------


def test_orpo_lambda_zero_refused_because_a_zero_weight_component_fails_coverage() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        ORPOLoss(lambda_=0.0)
    message = str(exc_info.value)
    assert "lambda_=0.0" in message
    assert "LossComponentCoverageGate" in message
    assert "always declares both" in message


def test_orpo_lambda_negative_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        ORPOLoss(lambda_=-1.0)
    assert "lambda_=-1.0" in str(exc_info.value)


def test_orpo_component_names_must_not_collide() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        ORPOLoss(component_name="shared", sft_component_name="shared")
    message = str(exc_info.value)
    assert "'shared'" in message
    assert "coverage denominator" in message


@pytest.mark.parametrize("field", PAIR_NAME_FIELDS)
@pytest.mark.parametrize("bad", ("", 7))
def test_orpo_every_name_field_refuses_a_non_name(field: str, bad: object) -> None:
    bad_kwargs: dict[str, Any] = {field: bad}
    with pytest.raises(LossConfigRefusal) as exc_info:
        ORPOLoss(**bad_kwargs)
    message = str(exc_info.value)
    assert field in message
    assert "not a name" in message


# ---------------------------------------------------------------------------
# Configuration refusals -- SimPO.
# ---------------------------------------------------------------------------


def test_simpo_beta_zero_inverts_the_signal_and_is_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        SimPOLoss(beta=0.0)
    message = str(exc_info.value)
    assert "beta=0.0" in message
    assert "preference signal" in message


def test_simpo_gamma_negative_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        SimPOLoss(gamma=-0.5)
    message = str(exc_info.value)
    assert "gamma=-0.5" in message
    assert "non-negative" in message


def test_simpo_gamma_nan_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        SimPOLoss(gamma=float("nan"))
    assert "gamma=nan" in str(exc_info.value)


def test_simpo_weight_zero_refused_because_it_would_fail_coverage() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        SimPOLoss(weight=0.0)
    message = str(exc_info.value)
    assert "weight=0.0" in message
    assert "LossComponentCoverageGate" in message


@pytest.mark.parametrize("field", SIMPO_NAME_FIELDS)
@pytest.mark.parametrize("bad", ("", 7))
def test_simpo_every_name_field_refuses_a_non_name(field: str, bad: object) -> None:
    bad_kwargs: dict[str, Any] = {field: bad}
    with pytest.raises(LossConfigRefusal) as exc_info:
        SimPOLoss(**bad_kwargs)
    message = str(exc_info.value)
    assert field in message
    assert "not a name" in message


# ---------------------------------------------------------------------------
# Configuration refusals -- CPO.
# ---------------------------------------------------------------------------


def test_cpo_beta_zero_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        CPOLoss(beta=0.0)
    assert "beta=0.0" in str(exc_info.value)


def test_cpo_lambda_zero_refused_because_the_sft_term_is_always_declared() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        CPOLoss(lambda_=0.0)
    message = str(exc_info.value)
    assert "lambda_=0.0" in message
    assert "SFT component" in message
    assert "LossComponentCoverageGate" in message


def test_cpo_lambda_negative_refused() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        CPOLoss(lambda_=-0.25)
    assert "lambda_=-0.25" in str(exc_info.value)


def test_cpo_component_names_must_not_collide() -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        CPOLoss(component_name="shared", sft_component_name="shared")
    message = str(exc_info.value)
    assert "'shared'" in message
    assert "coverage denominator" in message


@pytest.mark.parametrize("field", PAIR_NAME_FIELDS)
@pytest.mark.parametrize("bad", ("", 7))
def test_cpo_every_name_field_refuses_a_non_name(field: str, bad: object) -> None:
    bad_kwargs: dict[str, Any] = {field: bad}
    with pytest.raises(LossConfigRefusal) as exc_info:
        CPOLoss(**bad_kwargs)
    message = str(exc_info.value)
    assert field in message
    assert "not a name" in message


@pytest.mark.parametrize(
    "loss_cls",
    (ORPOLoss, SimPOLoss, CPOLoss),
    ids=("ORPO", "SimPO", "CPO"),
)
@pytest.mark.parametrize("ceiling", (0.0, -1.0, float("nan")))
def test_margin_metric_ceiling_must_be_strictly_positive(
    loss_cls: type[ReferenceFreeLoss],
    ceiling: float,
) -> None:
    with pytest.raises(LossConfigRefusal) as exc_info:
        loss_cls(margin_metric_ceiling=ceiling)
    message = str(exc_info.value)
    assert f"margin_metric_ceiling={ceiling!r}" in message
    assert "strictly positive" in message


# ---------------------------------------------------------------------------
# Interface surface: reference_free, required_columns, semantics, declaration.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loss_fn",
    (ORPOLoss(), SimPOLoss(), CPOLoss()),
    ids=("ORPO", "SimPO", "CPO"),
)
def test_reference_free_property_is_true(loss_fn: ReferenceFreeLoss) -> None:
    assert loss_fn.reference_free is True


@pytest.mark.parametrize(
    "loss_fn",
    (ORPOLoss(), SimPOLoss(), CPOLoss()),
    ids=("ORPO", "SimPO", "CPO"),
)
def test_semantics_is_reference_free_with_four_abstentions(
    loss_fn: ReferenceFreeLoss,
) -> None:
    semantics = loss_fn.semantics()
    assert semantics.reference_free is True
    assert semantics.ratio_scope is None
    assert semantics.clip_bounds is None
    assert semantics.kl_estimator is None
    assert semantics.group_size is None


@pytest.mark.parametrize(
    "loss_fn",
    (ORPOLoss(), SimPOLoss(), CPOLoss()),
    ids=("ORPO", "SimPO", "CPO"),
)
def test_required_columns_is_exactly_the_two_masks(loss_fn: ReferenceFreeLoss) -> None:
    # Reference-free means the schema carries ONLY the masks; a 4-tuple here
    # would be the anchored shape and is wrong for all three objectives.
    assert loss_fn.required_columns == ("chosen_loss_mask", "rejected_loss_mask")
    assert len(loss_fn.required_columns) == 2


@pytest.mark.parametrize(
    "loss_cls",
    (ORPOLoss, SimPOLoss, CPOLoss),
    ids=("ORPO", "SimPO", "CPO"),
)
def test_required_columns_track_the_custom_column_names(
    loss_cls: type[ReferenceFreeLoss],
) -> None:
    loss_fn = loss_cls(chosen_mask_column="c_mask", rejected_mask_column="r_mask")
    assert loss_fn.required_columns == ("c_mask", "r_mask")


def test_orpo_declaration_declares_both_components_and_pinned_metrics() -> None:
    declaration = ORPOLoss().declaration()
    assert declaration.components == ("sft_loss", "orpo_odds_loss")
    assert len(declaration.components) == 2
    accuracy, margin = declaration.metrics
    assert accuracy.name == "accuracy"
    assert accuracy.low == 0.0
    assert accuracy.high == 1.0
    assert accuracy.degenerate == (0.0,)
    assert margin.name == "orpo_margin_mean"
    assert margin.low == -100.0
    assert margin.high == 100.0
    assert margin.degenerate == ()


def test_orpo_margin_bounds_follow_the_configured_ceiling() -> None:
    declaration = ORPOLoss(margin_metric_ceiling=5.0).declaration()
    margin = declaration.metrics[1]
    assert margin.low == -5.0
    assert margin.high == 5.0


def test_simpo_declaration_declares_one_component_with_pinned_metrics() -> None:
    declaration = SimPOLoss(margin_metric_ceiling=5.0).declaration()
    assert declaration.components == ("simpo_loss",)
    accuracy, margin = declaration.metrics
    assert accuracy.name == "accuracy"
    assert accuracy.low == 0.0
    assert accuracy.high == 1.0
    assert accuracy.degenerate == ()
    assert margin.name == "simpo_margin_mean"
    assert margin.low == -5.0
    assert margin.high == 5.0
    assert margin.degenerate == ()


def test_cpo_declaration_declares_both_components_in_formula_order() -> None:
    declaration = CPOLoss(margin_metric_ceiling=5.0).declaration()
    assert declaration.components == ("cpo_loss", "sft_loss")
    assert len(declaration.components) == 2
    accuracy, margin = declaration.metrics
    assert accuracy.name == "accuracy"
    assert accuracy.low == 0.0
    assert accuracy.high == 1.0
    assert accuracy.degenerate == (0.0,)
    assert margin.name == "cpo_margin_mean"
    assert margin.low == -5.0
    assert margin.high == 5.0
    assert margin.degenerate == ()


def test_simpo_accuracy_degenerate_is_empty_where_orpo_and_cpo_pin_zero() -> None:
    # SimPO accuracy counts pairs already PAST the target margin, so a fresh
    # model legitimately opens at 0.0 with the gates running at STEP_ZERO;
    # declaring 0.0 degenerate would fail every healthy SimPO launch. ORPO and
    # CPO measure the plain ranking, where pinned-at-zero accuracy IS the
    # broken reading. Harmonising either side of this asymmetry is a bug.
    assert SimPOLoss().declaration().metrics[0].degenerate == ()
    assert ORPOLoss().declaration().metrics[0].degenerate == (0.0,)
    assert CPOLoss().declaration().metrics[0].degenerate == (0.0,)


# ---------------------------------------------------------------------------
# ORPO numerics.
# ---------------------------------------------------------------------------


def test_orpo_full_loss_pinned_component_by_component() -> None:
    # Row 0: chosen lps (-0.5,) under mask (1,) -> sum -0.5, count 1, average -0.5;
    #        rejected lps (-1.0,) under mask (1,) -> average -1.0.
    # Row 1: both sides lps (-0.2, -0.2) under mask (1, 1) -> average -0.2 each.
    # log_odds(avg) = avg - log(1 - exp(avg)), computed via expm1, so:
    #   row 0: log_odds_c ~= 0.4328, log_odds_r ~= -0.5413, margin ~= 0.9741 > 0;
    #   row 1: margin is exactly 0.0 and -log sigmoid(0.0) == log 2. The tie
    #          counts as NOT correct, so accuracy is 1/2 over the two rows.
    # SFT component (weight 1.0): -((-0.5) + (-0.4)) / (1 + 2) == 0.9 / 3 == 0.3.
    # Odds component (lambda_ == 0.5): 0.5 * (nls(margin0) + log 2) / 2.
    rows = [((-0.5,), (-1.0,)), ((-0.2, -0.2), (-0.2, -0.2))]
    batch = _paired_batch(((1,), (1, 1)), ((1,), (1, 1)))
    margin_row0 = (-0.5 - math.log(-math.expm1(-0.5))) - (-1.0 - math.log(-math.expm1(-1.0)))
    expected_odds = 0.5 * (math.log1p(math.exp(-margin_row0)) + math.log(2.0)) / 2.0
    out = ORPOLoss(lambda_=0.5)(_forward(rows), batch)
    sft, odds = out.components
    assert len(out.components) == 2
    assert sft.name == "sft_loss"
    assert sft.weight == 1.0
    assert sft.observed is True
    assert sft.contribution == pytest.approx(0.3)
    assert odds.name == "orpo_odds_loss"
    assert odds.weight == 0.5
    assert odds.observed is True
    assert odds.contribution == pytest.approx(expected_odds)
    assert out.loss == pytest.approx(0.3 + expected_odds)
    accuracy, margin_metric = out.metrics
    assert accuracy == MetricObservation(name="accuracy", value=0.5)
    assert margin_metric.name == "orpo_margin_mean"
    assert margin_metric.value == pytest.approx(margin_row0 / 2.0)


def test_orpo_probability_exactly_one_wall_refused_on_the_chosen_side() -> None:
    # A chosen completion whose length-normalised log-probability is exactly
    # 0.0 has p == 1.0 in float64: 1 - p == 0.0, the odds are undefined, and
    # the module must REFUSE (row and side named) rather than emit inf.
    rows = [((0.0,), (-0.5,))]
    batch = _paired_batch(((1,),), ((1,),))
    with pytest.raises(BatchRefusal) as exc_info:
        ORPOLoss()(_forward(rows), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "'chosen'" in message
    assert "exactly 1.0" in message
    assert "infinite margin" in message


def test_orpo_probability_exactly_one_wall_refused_on_the_rejected_side() -> None:
    rows = [((-0.5,), (0.0,))]
    batch = _paired_batch(((1,),), ((1,),))
    with pytest.raises(BatchRefusal) as exc_info:
        ORPOLoss()(_forward(rows), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "'rejected'" in message
    assert "exactly 1.0" in message
    assert "infinite margin" in message


def test_orpo_boolean_mask_entries_are_read_as_zero_and_one() -> None:
    # chosen lps (-0.3, -0.9) under mask (True, False) supervises ONLY position 0:
    #   sum -0.3, average -0.3. rejected lps (-0.7,) under (1,): average -0.7.
    # margin = log_odds(-0.3) - log_odds(-0.7) > 0; SFT term = -(-0.3 / 1) == 0.3.
    rows = [((-0.3, -0.9), (-0.7,))]
    batch = _paired_batch(((True, False),), ((1,),))
    margin = (-0.3 - math.log(-math.expm1(-0.3))) - (-0.7 - math.log(-math.expm1(-0.7)))
    expected = 0.3 + math.log1p(math.exp(-margin))
    out = ORPOLoss()(_forward(rows), batch)
    assert out.loss == pytest.approx(expected)
    assert out.metrics[0].value == 1.0


# ---------------------------------------------------------------------------
# SimPO numerics.
# ---------------------------------------------------------------------------


def test_simpo_gamma_zero_loss_hand_computed() -> None:
    # beta == 1, gamma == 0, one pair:
    #   chosen lps (-0.2, -0.3) under (1, 1): r_c = (1 / 2) * -0.5 == -0.25.
    #   rejected lps (-1.0,) under (1,): r_r = (1 / 1) * -1.0 == -1.0.
    #   margin == -0.25 - (-1.0) - 0.0 == 0.75; loss == log1p(exp(-0.75)).
    rows = [((-0.2, -0.3), (-1.0,))]
    batch = _paired_batch(((1, 1),), ((1,),))
    out = SimPOLoss()(_forward(rows), batch)
    assert out.loss == pytest.approx(math.log1p(math.exp(-0.75)))
    (component,) = out.components
    assert component.name == "simpo_loss"
    assert component.weight == 1.0
    assert component.observed is True
    assert component.contribution == out.loss
    accuracy, margin_metric = out.metrics
    assert accuracy == MetricObservation(name="accuracy", value=1.0)
    assert margin_metric.name == "simpo_margin_mean"
    assert margin_metric.value == pytest.approx(0.75)


def test_simpo_nonzero_gamma_subtracts_the_full_margin() -> None:
    # Same pair with gamma == 1.0: margin == 0.75 - 1.0 == -0.25.
    # loss == -log sigmoid(-0.25) == 0.25 + log1p(exp(-0.25)) (stable branch).
    # Accuracy counts the FULL margin: it is 0.0 here, and the loss must NOT
    # refuse -- a fresh model honestly starts below the target margin.
    rows = [((-0.2, -0.3), (-1.0,))]
    batch = _paired_batch(((1, 1),), ((1,),))
    out = SimPOLoss(gamma=1.0)(_forward(rows), batch)
    assert out.loss == pytest.approx(0.25 + math.log1p(math.exp(-0.25)))
    accuracy, margin_metric = out.metrics
    assert accuracy == MetricObservation(name="accuracy", value=0.0)
    assert margin_metric.value == pytest.approx(-0.25)


def test_simpo_weight_scales_the_component_and_the_scalar() -> None:
    rows = [((-0.2, -0.3), (-1.0,))]
    batch = _paired_batch(((1, 1),), ((1,),))
    out = SimPOLoss(weight=2.0)(_forward(rows), batch)
    expected = 2.0 * math.log1p(math.exp(-0.75))
    assert out.loss == pytest.approx(expected)
    assert out.components[0].weight == 2.0
    assert out.components[0].contribution == pytest.approx(expected)


def test_simpo_normalising_count_is_derived_from_the_mask() -> None:
    # IDENTICAL log-probabilities, DIFFERENT masks: if |y| came from a column
    # (or the raw completion length) instead of the mask, the two losses would
    # be equal.
    #   Batch A, chosen mask (1, 1, 1): r_c = (1 / 3) * (-1.5) == -0.5; margin 0.5.
    #   Batch B, chosen mask (1, 1, 0): r_c = (1 / 2) * (-0.6) == -0.3; margin 0.7.
    #   Both: rejected mask (1, 1, 1): r_r = (1 / 3) * (-3.0) == -1.0.
    rows = [((-0.1, -0.5, -0.9), (-1.0, -1.0, -1.0))]
    loss_fn = SimPOLoss()
    loss_a = loss_fn(_forward(rows), _paired_batch(((1, 1, 1),), ((1, 1, 1),))).loss
    loss_b = loss_fn(_forward(rows), _paired_batch(((1, 1, 0),), ((1, 1, 1),))).loss
    assert loss_a == pytest.approx(math.log1p(math.exp(-0.5)))
    assert loss_b == pytest.approx(math.log1p(math.exp(-0.7)))
    assert loss_a != pytest.approx(loss_b, rel=1e-9)


# ---------------------------------------------------------------------------
# CPO numerics -- and proof that no reference is read.
# ---------------------------------------------------------------------------


def test_cpo_full_loss_pinned_component_by_component() -> None:
    # One pair, beta == 0.1, lambda_ == 1.0, every token supervised:
    #   chosen lps (-0.2, -0.6) -> pi_c == -0.8; rejected lps (-1.0, -0.5)
    #   -> pi_r == -1.5. margin == 0.1 * -0.8 - 0.1 * -1.5 == 0.07 > 0.
    #   preference component: weight 1.0, contribution log1p(exp(-0.07)).
    #   SFT component: weight lambda_ == 1.0, contribution -(-0.8 / 2) == 0.4.
    rows = [((-0.2, -0.6), (-1.0, -0.5))]
    batch = _paired_batch(((1, 1),), ((1, 1),))
    margin = 0.1 * -0.8 - 0.1 * -1.5
    out = CPOLoss()(_forward(rows), batch)
    preference, sft = out.components
    assert preference.name == "cpo_loss"
    assert preference.weight == 1.0
    assert preference.contribution == pytest.approx(math.log1p(math.exp(-margin)))
    assert sft.name == "sft_loss"
    assert sft.weight == 1.0
    assert sft.contribution == pytest.approx(0.4)
    assert out.loss == pytest.approx(math.log1p(math.exp(-margin)) + 0.4)
    accuracy, margin_metric = out.metrics
    assert accuracy == MetricObservation(name="accuracy", value=1.0)
    assert margin_metric.name == "cpo_margin_mean"
    assert margin_metric.value == pytest.approx(margin)


def test_cpo_beta_and_lambda_enter_the_computation() -> None:
    # Same pair with beta == 2.0, lambda_ == 0.5:
    #   margin == 2.0 * -0.8 - 2.0 * -1.5 == 1.4;
    #   SFT contribution == 0.5 * 0.4 == 0.2, at declared weight 0.5.
    rows = [((-0.2, -0.6), (-1.0, -0.5))]
    batch = _paired_batch(((1, 1),), ((1, 1),))
    out = CPOLoss(beta=2.0, lambda_=0.5)(_forward(rows), batch)
    margin = 2.0 * -0.8 - 2.0 * -1.5
    assert out.components[0].contribution == pytest.approx(math.log1p(math.exp(-margin)))
    assert out.components[1].weight == 0.5
    assert out.components[1].contribution == pytest.approx(0.2)
    assert out.loss == pytest.approx(math.log1p(math.exp(-margin)) + 0.2)
    assert out.metrics[0].value == 1.0
    assert out.metrics[1].value == pytest.approx(margin)


def test_cpo_loss_is_unchanged_when_reference_columns_are_added() -> None:
    # Wildly different reference scores on top of the same pair: CPO drops the
    # reference entirely, so the loss must be bit-identical. A mutant that
    # subtracted reference margins would fail the equality outright.
    rows = [((-0.2, -0.6), (-1.0, -0.5))]
    plain = _paired_batch(((1, 1),), ((1, 1),))
    with_references = _paired_batch(
        ((1, 1),),
        ((1, 1),),
        extra_columns={
            "reference_chosen_logprob": (99.0,),
            "reference_rejected_logprob": (-50.0,),
        },
    )
    loss_fn = CPOLoss()
    plain_out = loss_fn(_forward(rows), plain)
    reference_out = loss_fn(_forward(rows), with_references)
    assert reference_out.loss == plain_out.loss
    assert [c.contribution for c in reference_out.components] == [
        c.contribution for c in plain_out.components
    ]


# ---------------------------------------------------------------------------
# Runtime refusals shared across the three objectives.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("loss_fn", "class_name"),
    ((ORPOLoss(), "ORPOLoss"), (SimPOLoss(), "SimPOLoss"), (CPOLoss(), "CPOLoss")),
)
def test_missing_rejected_mask_column_refused(
    loss_fn: ReferenceFreeLoss,
    class_name: str,
) -> None:
    batch = ExperienceBatch(columns={"chosen_loss_mask": ((1, 1), (1, 1))})
    with pytest.raises(BatchRefusal) as exc_info:
        loss_fn(_two_row_forward(), batch)
    message = str(exc_info.value)
    assert class_name in message
    assert "'rejected_loss_mask'" in message


@pytest.mark.parametrize(
    ("loss_fn", "class_name"),
    ((ORPOLoss(), "ORPOLoss"), (SimPOLoss(), "SimPOLoss"), (CPOLoss(), "CPOLoss")),
)
def test_zero_row_batch_refused_as_unmeasured(
    loss_fn: ReferenceFreeLoss,
    class_name: str,
) -> None:
    batch = _paired_batch((), ())
    with pytest.raises(BatchRefusal) as exc_info:
        loss_fn(_forward([]), batch)
    message = str(exc_info.value)
    assert class_name in message
    assert "0 rows" in message
    assert "unmeasurable" in message


@pytest.mark.parametrize(
    "loss_fn",
    (ORPOLoss(), SimPOLoss(), CPOLoss()),
    ids=("ORPO", "SimPO", "CPO"),
)
def test_non_finite_supervised_log_probability_refused(loss_fn: ReferenceFreeLoss) -> None:
    rows = [((-0.1, float("nan")), (-0.5, -0.5))]
    batch = _paired_batch(((1, 1),), ((1, 1),))
    with pytest.raises(BatchRefusal) as exc_info:
        loss_fn(_forward(rows), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "position 1" in message
    assert "nan" in message


def test_non_finite_at_an_unsupervised_position_is_masked_off() -> None:
    # The inf sits at a masked-off position: only -0.4 is supervised, so no
    # finiteness check runs there. margin = -0.4 - (-0.8) = 0.4.
    rows = [((float("inf"), -0.4), (-0.8,))]
    batch = _paired_batch(((0, 1),), ((1,),))
    out = SimPOLoss()(_forward(rows), batch)
    assert math.isfinite(out.loss)
    assert out.loss == pytest.approx(math.log1p(math.exp(-0.4)))


def test_orpo_zero_supervised_rejected_side_refused_as_unmeasurable() -> None:
    rows = [((-0.1, -0.2), (-0.5, -0.6))]
    batch = _paired_batch(((1, 1),), ((0, 0),))
    with pytest.raises(SupervisionRefusal) as exc_info:
        ORPOLoss()(_forward(rows), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "'rejected'" in message
    assert "0 supervised tokens" in message
    assert "normalising" in message


def test_simpo_zero_supervised_row_refused_before_any_division() -> None:
    rows = [((-0.1, -0.2), (-0.5, -0.6))]
    batch = _paired_batch(((0, 0),), ((1, 1),))
    with pytest.raises(SupervisionRefusal) as exc_info:
        SimPOLoss()(_forward(rows), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "'chosen'" in message
    assert "0 supervised tokens" in message
    assert "division by zero" in message


def test_cpo_zero_supervised_chosen_side_refused_as_unmeasurable() -> None:
    rows = [((-0.1, -0.2), (-0.5, -0.6))]
    batch = _paired_batch(((0, 0),), ((1, 1),))
    with pytest.raises(SupervisionRefusal) as exc_info:
        CPOLoss()(_forward(rows), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "'chosen'" in message
    assert "0 supervised tokens" in message
    assert "empty completion" in message


def test_orpo_mask_length_disagreeing_with_token_count_refused() -> None:
    rows = [((-0.1, -0.2), (-0.5, -0.6))]
    batch = _paired_batch(((1,),), ((1, 1),))
    with pytest.raises(BatchRefusal) as exc_info:
        ORPOLoss()(_forward(rows), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "'chosen'" in message
    assert "2" in message
    assert "1" in message


def test_simpo_wrong_forward_row_count_refused() -> None:
    rows = [((-0.1,), (-0.5,))]  # one row for a two-row batch
    with pytest.raises(BatchRefusal) as exc_info:
        SimPOLoss()(_forward(rows), _paired_batch())
    message = str(exc_info.value)
    assert "1" in message
    assert "2" in message


def test_cpo_row_of_three_is_not_a_pair() -> None:
    rows = [((-0.1,), (-0.5,), (-0.9,))]
    batch = _paired_batch(((1,),), ((1,),))
    with pytest.raises(BatchRefusal) as exc_info:
        CPOLoss()(_forward(rows), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "3" in message
    assert "expected 2" in message


def test_orpo_unsized_forward_row_refused() -> None:
    batch = _paired_batch(((1,),), ((1,),))
    with pytest.raises(BatchRefusal) as exc_info:
        ORPOLoss()(_forward([7]), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "int" in message


def test_simpo_fractional_mask_entry_refused() -> None:
    rows = [((-0.1, -0.2), (-0.5, -0.6))]
    batch = _paired_batch(((0.5, 1),), ((1, 1),))
    with pytest.raises(BatchRefusal) as exc_info:
        SimPOLoss()(_forward(rows), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "position 0" in message
    assert "0.5" in message
    assert "0 or 1" in message


def test_cpo_unconvertible_log_probability_refused() -> None:
    rows = [((object(), -0.2), (-0.5, -0.6))]
    batch = _paired_batch(((1, 1),), ((1, 1),))
    with pytest.raises(BatchRefusal) as exc_info:
        CPOLoss()(_forward(rows), batch)
    message = str(exc_info.value)
    assert "row 0" in message
    assert "position 0" in message
    assert "object" in message
