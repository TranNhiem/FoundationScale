"""Tests for PPOCompositeLoss: construction, declaration, and compute_with_report.

PPOAlgorithm and check_ppo_requirements are covered by a sibling module and are
deliberately untested here; this suite attacks only the composite -- its
construction refusals, its structural KL absence, the single-measurement value
seam, the supplied-values override, the priced LossOutput, and the retained
estimator report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from foundationscale.gates.objective_gates import MetricObservation
from foundationscale.rl import (
    BatchRefusal,
    ExperienceBatch,
    LossConfigRefusal,
    PPOClippedPolicyLoss,
    ValueFunctionLoss,
)
from foundationscale.rl.advantage import AdvantageResult, LearnedValueAdvantageEstimation
from foundationscale.rl.ppo import PPOCompositeLoss
from foundationscale.rl.ppo_objectives import KLPenaltyLoss
from foundationscale.rl.value_head import ValueCapabilities

# Two single-token rows, both terminated, both supervised. Unit policy-ratios
# (current == old == 0.0) make the clip inert, and a terminated single-token
# trajectory has advantage r - v under any gamma/lam convention, so the
# estimator's hyperparameters never enter the hand arithmetic below.
_NAME_FIELDS = ("reward_column", "prompt_id_column", "terminated_column")
_ESTIMATOR_COLUMNS = ("rewards", "prompt_ids", "terminated")


@dataclass(frozen=True)
class _CountingValueHead:
    """A ValueHead that records every reading it hands out.

    The nth reading is ((n,), (2n,)): each call returns DIFFERENT numbers, so a
    consumer quietly fed a second reading is visible in the priced scalar, not
    just in the counter.
    """

    readings: list[tuple[tuple[float, ...], ...]] = field(default_factory=list)

    def estimate(self, _batch: ExperienceBatch) -> Sequence[Sequence[float]]:
        ordinal = float(len(self.readings) + 1)
        reading: tuple[tuple[float, ...], ...] = ((ordinal,), (2.0 * ordinal,))
        self.readings.append(reading)
        return reading

    def capabilities(self) -> ValueCapabilities:
        return ValueCapabilities(
            granularity="token",
            shares_policy_trunk=False,
            reports_old_values=True,
        )


def _head() -> _CountingValueHead:
    return _CountingValueHead()


def _estimator() -> LearnedValueAdvantageEstimation:
    return LearnedValueAdvantageEstimation()


def _composite(**overrides: Any) -> PPOCompositeLoss:
    kwargs: dict[str, Any] = {"value_head": _head(), "advantage_fn": _estimator()}
    kwargs.update(overrides)
    return PPOCompositeLoss(**kwargs)


def _batch(
    *,
    drop: tuple[str, ...] = (),
    reward_column: str = "rewards",
    prompt_id_column: str = "prompt_ids",
    terminated_column: str = "terminated",
    advantages: tuple[tuple[float, ...], ...] = ((2.0,), (3.0,)),
    prompt_ids: tuple[str, ...] = ("p0", "p1"),
    terminated: tuple[bool, ...] = (True, True),
) -> ExperienceBatch:
    # rewards is ONE FINITE SCALAR PER ROW -- the terminal reward the estimator
    # lands on each row's last supervised token -- NOT a per-token sequence; the
    # estimator refuses a nested shape outright. The advantages ((2.0,), (3.0,))
    # below equal r - v for rewards (3.0, 5.0) and the head's first reading
    # ((1.0,), (2.0,)) -- the producer pairing the binding anchors but never
    # cross-checks.
    columns: dict[str, Any] = {
        "loss_mask": ((1,), (1,)),
        "old_logprobs": ((0.0,), (0.0,)),
        "advantages": advantages,
        "returns": ((3.0,), (5.0,)),
        reward_column: (3.0, 5.0),
        prompt_id_column: prompt_ids,
        terminated_column: terminated,
    }
    for name in drop:
        del columns[name]
    return ExperienceBatch(columns=columns)


def _good_forward(_batch: ExperienceBatch) -> Sequence[Sequence[float]]:
    return ((0.0,), (0.0,))


def _direct_advantage(
    estimator: LearnedValueAdvantageEstimation,
    *,
    prompt_ids: tuple[str, ...] = ("p0", "p1"),
    values: Sequence[Sequence[float]] = ((1.0,), (2.0,)),
    terminated: tuple[bool, ...] = (True, True),
) -> AdvantageResult:
    # Mirrors the composite's own estimator call: per-ROW scalar rewards
    # (3.0, 5.0), per-token mask and values, per-row termination flags.
    return estimator.compute(
        prompt_ids=prompt_ids,
        rewards=(3.0, 5.0),
        mask=((1,), (1,)),
        values=values,
        terminated=terminated,
    )


class TestCompositeConstructionRefusals:
    @pytest.mark.parametrize("field_name", _NAME_FIELDS)
    @pytest.mark.parametrize("bad", ["", 7])
    def test_every_column_name_field_refuses_a_non_name(self, field_name: str, bad: object) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            _composite(**{field_name: bad})
        message = str(exc_info.value)
        assert f"{field_name}={bad!r}" in message
        assert "non-empty strings" in message

    def test_none_value_head_refused(self) -> None:
        # The annotation already forbids None, yet the runtime refusal must
        # stay: untyped call sites are invisible to the typechecker, and
        # without this guard an absent head would surface later as an
        # AttributeError on .estimate inside compute. Not dead code -- do not
        # delete it.
        with pytest.raises(LossConfigRefusal) as exc_info:
            PPOCompositeLoss(value_head=None, advantage_fn=_estimator())
        message = str(exc_info.value)
        assert "value_head=None" in message
        assert "1 of 1 required inputs absent for ppo" in message

    def test_none_advantage_fn_refused(self) -> None:
        # Same non-optional-annotation story as the value head: the refusal
        # exists for callers the typechecker cannot see, because the temporal
        # estimator is a required role, not an optional extra.
        with pytest.raises(LossConfigRefusal) as exc_info:
            PPOCompositeLoss(value_head=_head(), advantage_fn=None)
        message = str(exc_info.value)
        assert "advantage_fn=None" in message
        assert "1 of 1 required inputs absent for ppo advantage estimation" in message

    def test_wrong_policy_slot_type_named(self) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            _composite(policy_loss=ValueFunctionLoss())
        message = str(exc_info.value)
        assert "1 of 1 policy slots must carry PPOClippedPolicyLoss" in message
        assert "ValueFunctionLoss" in message

    def test_wrong_value_slot_type_named(self) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            _composite(value_loss=PPOClippedPolicyLoss())
        message = str(exc_info.value)
        assert "1 of 1 value slots must carry ValueFunctionLoss" in message
        assert "PPOClippedPolicyLoss" in message

    def test_kl_substitute_refused_because_absence_is_none(self) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            _composite(kl_loss="kl")
        message = str(exc_info.value)
        assert "absence is None, never a substitute" in message
        assert "str" in message

    def test_non_head_refused(self) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            _composite(value_head=object())
        message = str(exc_info.value)
        assert "must carry a ValueHead" in message
        assert "object" in message

    def test_non_estimator_advantage_fn_refused(self) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            _composite(advantage_fn=object())
        message = str(exc_info.value)
        assert "LearnedValueAdvantageEstimation" in message
        assert "not an undeclared substitute" in message

    def test_duplicate_component_refused_with_both_contributors_named(self) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            _composite(policy_loss=PPOClippedPolicyLoss(component_name="value_loss"))
        message = str(exc_info.value)
        assert "'value_loss'" in message
        assert "both 'policy_loss' and 'value_loss'" in message
        assert "2 of 2 sub-losses" in message

    def test_duplicate_component_count_moves_with_the_kl_term_wired(self) -> None:
        # The denominator in the refusal is the number of contributing
        # sub-losses: wiring a KL term must move it 2 -> 3.
        with pytest.raises(LossConfigRefusal) as exc_info:
            _composite(
                value_loss=ValueFunctionLoss(component_name="kl_penalty"),
                kl_loss=KLPenaltyLoss(),
            )
        message = str(exc_info.value)
        assert "'kl_penalty'" in message
        assert "both 'value_loss' and 'kl_loss'" in message
        assert "2 of 3 sub-losses" in message


class TestCompositeDeclarationStructure:
    def test_kl_absence_is_structural_not_a_zero_coefficient(self) -> None:
        # A test that only priced the loss scalar could not distinguish "no KL
        # term" from "KL term with zero weight": KLPenaltyLoss refuses a zero
        # coefficient internally, so absence must be visible in the DECLARATION
        # -- one fewer component, by name, and one fewer metric, by name.
        absent = _composite().declaration()
        present = _composite(kl_loss=KLPenaltyLoss()).declaration()
        assert absent.components == ("ppo_policy_loss", "value_loss")
        assert present.components == ("ppo_policy_loss", "value_loss", "kl_penalty")
        assert len(present.components) == len(absent.components) + 1
        assert tuple(metric.name for metric in absent.metrics) == ("clip_fraction",)
        assert tuple(metric.name for metric in present.metrics) == (
            "clip_fraction",
            "kl_estimate",
        )

    def test_semantics_track_the_kl_term(self) -> None:
        absent = _composite().semantics()
        present = _composite(kl_loss=KLPenaltyLoss()).semantics()
        assert absent.kl_estimator is None
        assert absent.reference_free is True
        assert present.kl_estimator == "k3"
        assert present.reference_free is False
        assert absent.clip_bounds == (0.8, 1.2)
        assert present.clip_bounds == (0.8, 1.2)
        assert absent.ratio_scope == "token"
        assert absent.group_size is None


class TestCompositeMissingEstimatorColumns:
    @pytest.mark.parametrize("column", _ESTIMATOR_COLUMNS)
    def test_each_absent_column_is_named_with_the_count(self, column: str) -> None:
        with pytest.raises(BatchRefusal) as exc_info:
            _composite().compute_with_report(_good_forward, _batch(drop=(column,)))
        message = str(exc_info.value)
        assert "1 of 3" in message
        assert f"{{{column}}}" in message

    def test_two_absent_columns_move_the_count_and_name_both(self) -> None:
        with pytest.raises(BatchRefusal) as exc_info:
            _composite().compute_with_report(_good_forward, _batch(drop=("rewards", "terminated")))
        message = str(exc_info.value)
        assert "2 of 3" in message
        assert "{rewards, terminated}" in message

    def test_configured_names_feed_the_absence_check_not_the_defaults(self) -> None:
        # The batch carries the default "rewards" column; the composite reads
        # "r", and the refusal must name the OFFENDING configured name.
        composite = _composite(reward_column="r")
        with pytest.raises(BatchRefusal) as exc_info:
            composite.compute_with_report(_good_forward, _batch())
        message = str(exc_info.value)
        assert "1 of 3" in message
        assert "{r}" in message

    def test_configured_columns_are_read_under_their_configured_names(self) -> None:
        # Success with the default estimator columns absent proves the reads
        # go through the configured names, not the defaults.
        composite = _composite(reward_column="r", prompt_id_column="p", terminated_column="t")
        output, retained = composite.compute_with_report(
            _good_forward,
            _batch(reward_column="r", prompt_id_column="p", terminated_column="t"),
        )
        assert output.loss == pytest.approx(4.0)
        assert retained == _direct_advantage(composite.advantage_fn)


class TestSingleMeasuredReading:
    def test_the_estimate_is_measured_once_and_shared_by_both_consumers(self) -> None:
        # The first (and only permitted) reading is ((1.0,), (2.0,)).
        # Policy: unit ratios, inert clip, terms 2.0 and 3.0 -> -(2.0+3.0)/2 = -2.5.
        # Value: (1.0-3.0)^2 = 4.0 and (2.0-5.0)^2 = 9.0 -> 13.0/2 = 6.5.
        # Composite scalar: -2.5 + 6.5 = 4.0. Had a second reading ((2.0,),
        # (4.0,)) been taken for the value regression the errors would both be
        # 1.0 and the scalar would read 1.0 - 2.5 = -1.5; had the ESTIMATOR
        # been fed it, retained would not equal the direct compute below.
        head = _head()
        composite = _composite(value_head=head)
        output, retained = composite.compute_with_report(_good_forward, _batch())
        assert head.readings == [((1.0,), (2.0,))]
        assert output.loss == pytest.approx(4.0)
        assert output.loss != pytest.approx(-1.5)
        assert retained == _direct_advantage(composite.advantage_fn)

    def test_call_prices_through_one_reading_as_well(self) -> None:
        head = _head()
        composite = _composite(value_head=head)
        output = composite(_good_forward, _batch())
        assert len(head.readings) == 1
        assert output.loss == pytest.approx(4.0)
        assert tuple(component.name for component in output.components) == (
            "ppo_policy_loss",
            "value_loss",
        )

    def test_supplied_values_bypass_the_head_and_feed_both_consumers(self) -> None:
        # Supplied values ((0.5,), (0.25,)) give advantages r - v = 2.5, 4.75
        # (mirrored in the batch's producer column below):
        # policy = -(2.5 + 4.75)/2 = -3.625;
        # value errors (0.5-3.0)^2 = 6.25 and (0.25-5.0)^2 = 22.5625 ->
        #   28.8125/2 = 14.40625;
        # scalar = -3.625 + 14.40625 = 10.78125.
        head = _head()
        composite = _composite(value_head=head)
        output, retained = composite.compute_with_report(
            _good_forward,
            _batch(advantages=((2.5,), (4.75,))),
            values=((0.5,), (0.25,)),
        )
        assert head.readings == []
        assert output.loss == pytest.approx(10.78125)
        assert retained == _direct_advantage(composite.advantage_fn, values=((0.5,), (0.25,)))


class TestCompositeLossOutput:
    def test_every_component_reports_weight_observed_and_contribution(self) -> None:
        # Arithmetic as above: policy contributes exactly -2.5, value exactly
        # 6.5, weight 1.0 each, and the composite adds NO weighting layer of
        # its own, so the scalar is the plain sum, 4.0.
        output, _retained = _composite().compute_with_report(_good_forward, _batch())
        assert output.loss == pytest.approx(4.0)
        policy_component, value_component = output.components
        assert policy_component.name == "ppo_policy_loss"
        assert policy_component.weight == 1.0
        assert policy_component.observed is True
        assert policy_component.contribution == pytest.approx(-2.5)
        assert value_component.name == "value_loss"
        assert value_component.weight == 1.0
        assert value_component.observed is True
        assert value_component.contribution == pytest.approx(6.5)
        assert output.metrics == (MetricObservation(name="clip_fraction", value=0.0),)


class TestCompositeReport:
    def test_the_denominators_are_the_estimators_not_a_recomputation(self) -> None:
        # RISK 2 RESOLVED FROM THE SOURCE: used/offered are NOT a trajectory
        # compaction. LearnedValueAdvantageEstimation excludes nothing --
        # prompt_ids participate only in the shared length check -- so
        # used == offered == len(batch) and rows is the identity subsequence,
        # by construction, for every batch. What keeps its teeth here is the
        # WEIGHTS and the full-result equality: a composite that grouped rows,
        # fabricated termination flags, or fed a second reading prices
        # different numbers and fails the exact comparisons.
        #
        # Arithmetic (gamma=1.0, lambda_=0.95 defaults; one supervised token
        # per row; the head's first reading feeds values ((1.0,), (2.0,))):
        #   row 0, terminated=False (truncated): the tail bootstraps from the
        #     row's own last value, so delta = r + gamma*V(t_last) - V(t_last)
        #     = 3.0 + 1.0*1.0 - 1.0 = 3.0 -- at gamma=1.0 the value terms
        #     cancel and the token carries the RAW reward. A fabricated
        #     all-True termination would read 3.0 - 1.0 = 2.0 here, so the
        #     exact weights below catch an invented flag too.
        #   row 1, terminated=True: delta = 5.0 + 1.0*0.0 - 2.0 = 3.0; A = 3.0.
        estimator = _estimator()
        composite = _composite(advantage_fn=estimator)
        batch = _batch(prompt_ids=("p0", "p0"), terminated=(False, True))
        _output, retained = composite.compute_with_report(_good_forward, batch)
        assert retained == _direct_advantage(
            estimator, prompt_ids=("p0", "p0"), terminated=(False, True)
        )
        assert retained.weights == ((3.0,), (3.0,))
        assert retained.offered == 2
        assert retained.offered == len(batch)
        assert retained.used == 2
        assert retained.used == retained.offered
        assert retained.rows == (0, 1)
        assert retained.method == "LearnedValueAdvantageEstimation"

    def test_offered_equals_the_batch_length_row_for_row(self) -> None:
        _output, retained = _composite().compute_with_report(_good_forward, _batch())
        assert retained.offered == 2
