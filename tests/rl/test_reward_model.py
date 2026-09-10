"""Tests for ``foundationscale.rl.reward_model``.

Covers the Bradley-Terry arithmetic on hand-computed margins (asserting the
exact numbers, not mere finiteness), ``required_columns``, ``declaration()``,
``semantics()`` and its abstained axes, every construction refusal, every
batch refusal, and ``check_reward_model_requirements`` on both a satisfied
wiring and each refusal path.

WHAT IS CLAIMED: the tests below exercise only objects and behaviours that
exist in the real module; ``__call__`` is driven through a duck-typed batch
(a plain tuple), since the loss only calls ``len(batch)`` and passes the
batch through to ``forward_fn`` untouched.

WHAT IS NOT CLAIMED: that ``ExperienceBatch`` construction, the reward
head's architecture, or the inference-side scoring path is tested here;
none of those live in this module.
"""

from __future__ import annotations

import math

import pytest

from foundationscale.gates.objective_gates import LossComponent, MetricObservation
from foundationscale.rl.interfaces import (
    BatchRefusal,
    LossConfigRefusal,
    LossOutput,
    SupervisionRefusal,
)
from foundationscale.rl.reward_model import (
    RewardModelLoss,
    check_reward_model_requirements,
)


def _scoring_forward(pairs: tuple[tuple[object, object], ...]):
    """Build a forward_fn returning the given (chosen, rejected) rows."""

    def forward_fn(batch: object) -> tuple[tuple[object, object], ...]:
        return pairs

    return forward_fn


class TestBradleyTerryArithmetic:
    def test_single_correct_pair_matches_hand_computed_softplus(self) -> None:
        # margin = log(2); loss = softplus(-log(2)) = log(1 + 1/2) = log(1.5).
        loss_fn = RewardModelLoss()
        out = loss_fn(_scoring_forward(((math.log(2.0), 0.0),)), ("row-0",))
        assert isinstance(out, LossOutput)
        assert out.loss == pytest.approx(math.log(1.5))

    def test_weight_scales_the_mean_pair_loss(self) -> None:
        loss_fn = RewardModelLoss(weight=2.0)
        out = loss_fn(_scoring_forward(((math.log(2.0), 0.0),)), ("row-0",))
        assert out.loss == pytest.approx(2.0 * math.log(1.5))

    def test_two_pairs_average_and_accuracy_is_counted(self) -> None:
        # margins: +log(2) -> log(1.5); -log(2) -> softplus(+log(2)) = log(3).
        loss_fn = RewardModelLoss()
        batch = ("row-0", "row-1")
        out = loss_fn(
            _scoring_forward(((math.log(2.0), 0.0), (0.0, math.log(2.0)))),
            batch,
        )
        expected = (math.log(1.5) + math.log(3.0)) / 2.0
        assert out.loss == pytest.approx(expected)
        (metric,) = out.metrics
        assert isinstance(metric, MetricObservation)
        assert metric.name == "accuracy"
        assert metric.value == pytest.approx(0.5)

    def test_component_and_metric_carry_declared_names(self) -> None:
        loss_fn = RewardModelLoss(component_name="rm", metric_name="acc")
        out = loss_fn(_scoring_forward(((2.0, 1.0),)), ("row-0",))
        (component,) = out.components
        assert isinstance(component, LossComponent)
        assert component.name == "rm"
        assert component.weight == pytest.approx(1.0)
        assert component.contribution == pytest.approx(out.loss)
        (metric,) = out.metrics
        assert metric.name == "acc"
        assert metric.value == pytest.approx(1.0)

    def test_large_negative_margin_does_not_underflow(self) -> None:
        # margin = -1000: naive log(sigmoid) underflows; softplus form = 1000.
        loss_fn = RewardModelLoss()
        out = loss_fn(_scoring_forward(((0.0, 1000.0),)), ("row-0",))
        assert out.loss == pytest.approx(1000.0)


class TestRequiredColumnsAndDeclaration:
    def test_required_columns_defaults_in_stable_order(self) -> None:
        assert RewardModelLoss().required_columns == ("chosen_reward", "rejected_reward")

    def test_required_columns_reflect_custom_names(self) -> None:
        loss_fn = RewardModelLoss(chosen_score_column="a", rejected_score_column="b")
        assert loss_fn.required_columns == ("a", "b")

    def test_reference_free_is_true(self) -> None:
        assert RewardModelLoss().reference_free is True

    def test_declaration_components_and_metric_bounds(self) -> None:
        decl = RewardModelLoss().declaration()
        assert decl.components == ("reward_model_loss",)
        (expectation,) = decl.metrics
        assert expectation.name == "accuracy"
        assert expectation.low == pytest.approx(0.0)
        assert expectation.high == pytest.approx(1.0)
        assert expectation.degenerate == (0.0,)


class TestSemantics:
    def test_reference_free_constrained_other_axes_abstain(self) -> None:
        semantics = RewardModelLoss().semantics()
        assert semantics.reference_free is True
        # The three axes a reward model does not speak. These are the REAL
        # AlgorithmSemantics fields; the names this originally asserted
        # (forms_importance_ratio, reads_group, clips_ratio) exist nowhere.
        assert semantics.group_size is None
        assert semantics.ratio_scope is None
        assert semantics.kl_estimator is None
        assert semantics.clip_bounds is None


class TestConstructionRefusals:
    def test_zero_weight_refused(self) -> None:
        with pytest.raises(LossConfigRefusal, match=r"weight=0\.0"):
            RewardModelLoss(weight=0.0)

    def test_non_finite_weight_refused(self) -> None:
        with pytest.raises(LossConfigRefusal, match=r"weight=inf"):
            RewardModelLoss(weight=math.inf)

    def test_bool_weight_refused(self) -> None:
        with pytest.raises(LossConfigRefusal, match=r"weight=True"):
            RewardModelLoss(weight=True)

    @pytest.mark.parametrize(
        "field",
        (
            "chosen_score_column",
            "rejected_score_column",
            "component_name",
            "metric_name",
        ),
    )
    def test_empty_name_refused_naming_the_field(self, field: str) -> None:
        with pytest.raises(LossConfigRefusal, match=field):
            RewardModelLoss(**{field: ""})

    def test_identical_score_columns_refused(self) -> None:
        with pytest.raises(
            LossConfigRefusal,
            match=r"chosen_score_column and rejected_score_column are both 'same'",
        ):
            RewardModelLoss(chosen_score_column="same", rejected_score_column="same")


class TestBatchRefusals:
    def test_row_count_mismatch_refused_with_both_counts(self) -> None:
        loss_fn = RewardModelLoss()
        with pytest.raises(BatchRefusal, match=r"returned 1 score rows for a batch of 2 rows"):
            loss_fn(_scoring_forward(((1.0, 0.0),)), ("row-0", "row-1"))

    def test_pair_with_wrong_width_refused_with_row_index_and_count(self) -> None:
        loss_fn = RewardModelLoss()
        with pytest.raises(BatchRefusal, match=r"row 1: forward_fn returned 3 scores"):
            loss_fn(
                _scoring_forward(((1.0, 0.0), (1.0, 0.0, 0.5))),
                ("row-0", "row-1"),
            )

    def test_non_finite_chosen_reward_refused(self) -> None:
        loss_fn = RewardModelLoss()
        with pytest.raises(
            BatchRefusal,
            match=r"chosen reward at row 0 is nan, which is not finite",
        ):
            loss_fn(_scoring_forward(((math.nan, 0.0),)), ("row-0",))

    def test_non_finite_rejected_reward_refused(self) -> None:
        loss_fn = RewardModelLoss()
        with pytest.raises(
            BatchRefusal,
            match=r"rejected reward at row 0 is inf, which is not finite",
        ):
            loss_fn(_scoring_forward(((1.0, math.inf),)), ("row-0",))

    def test_non_convertible_reward_refused_naming_member_and_row(self) -> None:
        loss_fn = RewardModelLoss()
        with pytest.raises(
            BatchRefusal,
            match=r"chosen reward at row 0 is 'x', which does not convert",
        ):
            loss_fn(_scoring_forward((("x", 0.0),)), ("row-0",))

    def test_zero_observed_pairs_refused_as_unmeasurable(self) -> None:
        loss_fn = RewardModelLoss()
        with pytest.raises(
            SupervisionRefusal,
            match=r"0 preference pairs across 0 batch rows",
        ):
            loss_fn(_scoring_forward(()), ())


class TestCheckRewardModelRequirements:
    def test_satisfied_wiring_passes(self) -> None:
        loss_fn = RewardModelLoss()
        check_reward_model_requirements(
            loss_fn, ("prompt", "chosen_reward", "rejected_reward", "extra")
        )
        check_reward_model_requirements(loss_fn, ("chosen_reward", "rejected_reward"))

    def test_missing_column_refused_with_count_and_names(self) -> None:
        loss_fn = RewardModelLoss()
        with pytest.raises(
            BatchRefusal,
            match=r"1 of 2 required score column\(s\) absent: \('rejected_reward',\)",
        ):
            check_reward_model_requirements(loss_fn, ("chosen_reward",))

    def test_both_columns_missing_refused_with_count(self) -> None:
        loss_fn = RewardModelLoss()
        with pytest.raises(BatchRefusal, match=r"2 of 2 required score column\(s\) absent"):
            check_reward_model_requirements(loss_fn, ("unrelated",))

    def test_missing_column_named_in_refusal(self) -> None:
        loss_fn = RewardModelLoss(chosen_score_column="c", rejected_score_column="r")
        with pytest.raises(BatchRefusal, match=r"\('r',\)"):
            check_reward_model_requirements(loss_fn, ("c",))
