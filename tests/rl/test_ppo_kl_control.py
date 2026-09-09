"""Tests for the PPO KL coefficient controllers and the k3 penalty loss."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
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
from foundationscale.rl.ppo_objectives import (
    AdaptiveKLController,
    FixedKLCoefficient,
    KLCoefficientController,
    KLPenaltyLoss,
)

# A non-trivial batch: two rows, four supervised tokens. The token at row 1
# position 0 is deliberately masked out while carrying a huge log-ratio, so a
# code path that forgets the mask or averages over the wrong denominator blows
# the hand-computed expectation up by orders of magnitude.
_MASKS = ((1, 1, 1), (0, 1))
_REFERENCES = ((0.0, -0.5, -1.0), (2.0, -0.25))
_CURRENT = ((-0.1, -0.3, -1.3), (-7.0, -0.25))


def _batch(masks: Any = _MASKS, references: Any = _REFERENCES) -> ExperienceBatch:
    return ExperienceBatch(columns={"loss_mask": masks, "reference_logprobs": references})


def _forward(rows: Any) -> ForwardFn:
    materialised = list(rows)
    return lambda _experience: list(materialised)


class _ZeroCoefficientController:
    """Presence-shaped duck whose coefficient is exactly zero."""

    def coefficient(self) -> float:
        return 0.0

    def update(self, _measured_kl: float) -> _ZeroCoefficientController:
        return self


class _WrongShapeController:
    """Method names match the protocol; the return shapes do not."""

    def coefficient(self) -> str:
        return "0.5"

    def update(self) -> None:
        pass


class _DecayedController:
    """Reads 0.5 once, then 0.0 on every later reading."""

    def __init__(self) -> None:
        self._calls = 0

    def coefficient(self) -> float:
        self._calls += 1
        if self._calls == 1:
            return 0.5
        return 0.0

    def update(self, _measured_kl: float) -> _DecayedController:
        return self


def test_fixed_coefficient_returns_the_value_it_was_built_with() -> None:
    assert FixedKLCoefficient().coefficient() == 0.04
    assert FixedKLCoefficient(value=0.7).coefficient() == 0.7


def test_fixed_update_returns_self_and_the_coefficient_never_moves() -> None:
    controller = FixedKLCoefficient(value=0.7)
    assert controller.update(0.0) is controller
    assert controller.update(1000.0) is controller
    assert isinstance(controller.update(0.02), FixedKLCoefficient)
    assert controller.coefficient() == 0.7


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        (0.0, "value=0.0"),
        (-1.0, "value=-1.0"),
        (True, "value=True"),
        (float("nan"), "value=nan"),
        (float("inf"), "value=inf"),
    ],
)
def test_fixed_value_must_be_finite_and_strictly_positive(value: object, fragment: str) -> None:
    with pytest.raises(LossConfigRefusal, match=fragment):
        FixedKLCoefficient(value=value)


def test_adaptive_coefficient_reports_the_value_in_force() -> None:
    assert AdaptiveKLController().coefficient() == 0.04
    assert AdaptiveKLController(value=0.09).coefficient() == 0.09


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        (0.0, "value=0.0"),
        (-0.04, "value=-0.04"),
        (True, "value=True"),
        (float("nan"), "value=nan"),
    ],
)
def test_adaptive_value_must_be_finite_and_strictly_positive(value: object, fragment: str) -> None:
    with pytest.raises(LossConfigRefusal, match=fragment):
        AdaptiveKLController(value=value)


@pytest.mark.parametrize(
    ("target_kl", "fragment"),
    [(0.0, "target_kl=0.0"), (-0.02, "target_kl=-0.02"), (True, "target_kl=True")],
)
def test_adaptive_target_kl_must_be_finite_and_strictly_positive(
    target_kl: object, fragment: str
) -> None:
    with pytest.raises(LossConfigRefusal, match=fragment):
        AdaptiveKLController(target_kl=target_kl)


@pytest.mark.parametrize(
    ("tolerance_band", "fragment"),
    [
        (1.0, "tolerance_band=1.0"),
        (0.9, "tolerance_band=0.9"),
        (True, "tolerance_band=True"),
    ],
)
def test_adaptive_tolerance_band_must_strictly_exceed_one(
    tolerance_band: object, fragment: str
) -> None:
    # Both ends of the guard: 1.0 itself has no dead-band interior, and below
    # it the band inverts. A >= flipped to > would pass the 1.0 leg.
    with pytest.raises(LossConfigRefusal, match=fragment):
        AdaptiveKLController(tolerance_band=tolerance_band)


@pytest.mark.parametrize(
    ("growth", "fragment"),
    [(1.0, "growth=1.0"), (0.75, "growth=0.75"), (True, "growth=True")],
)
def test_adaptive_growth_must_strictly_exceed_one(growth: object, fragment: str) -> None:
    with pytest.raises(LossConfigRefusal, match=fragment):
        AdaptiveKLController(growth=growth)


@pytest.mark.parametrize(
    ("decay", "fragment"),
    [(0.0, "decay=0.0"), (1.0, "decay=1.0"), (True, "decay=True")],
)
def test_adaptive_decay_must_lie_strictly_inside_the_unit_interval(
    decay: object, fragment: str
) -> None:
    # Both ends of (0.0, 1.0): neither may pass, for opposite reasons.
    with pytest.raises(LossConfigRefusal, match=fragment):
        AdaptiveKLController(decay=decay)


@pytest.mark.parametrize(
    ("minimum", "fragment"),
    [(0.0, "minimum=0.0"), (-0.001, "minimum=-0.001"), (True, "minimum=True")],
)
def test_adaptive_minimum_must_be_finite_and_strictly_positive(
    minimum: object, fragment: str
) -> None:
    with pytest.raises(LossConfigRefusal, match=fragment):
        AdaptiveKLController(minimum=minimum)


@pytest.mark.parametrize(
    ("maximum", "fragment"),
    [(float("inf"), "maximum=inf"), (True, "maximum=True")],
)
def test_adaptive_maximum_must_be_finite_when_set(maximum: object, fragment: str) -> None:
    with pytest.raises(LossConfigRefusal, match=fragment):
        AdaptiveKLController(maximum=maximum)


def test_adaptive_maximum_must_be_strictly_above_minimum() -> None:
    # Equality is the off-by-one case: a >= flipped to > would admit it.
    with pytest.raises(LossConfigRefusal, match="strictly above minimum=0.1"):
        AdaptiveKLController(minimum=0.1, maximum=0.1)
    with pytest.raises(LossConfigRefusal, match="strictly above minimum=0.1"):
        AdaptiveKLController(minimum=0.1, maximum=0.05)


def test_adaptive_value_above_the_stated_ceiling_refused() -> None:
    with pytest.raises(LossConfigRefusal, match="exceeds the stated ceiling"):
        AdaptiveKLController(value=0.1, maximum=0.05)


def test_adaptive_value_below_the_stated_floor_refused() -> None:
    with pytest.raises(LossConfigRefusal, match="lies below the stated floor"):
        AdaptiveKLController(value=1.0e-9)


@pytest.mark.parametrize("controller", [FixedKLCoefficient(), AdaptiveKLController()])
@pytest.mark.parametrize(
    ("reading", "fragment"),
    [
        (float("nan"), "KL nan"),
        (float("inf"), "KL inf"),
        (-0.5, "KL -0.5"),
    ],
)
def test_update_refuses_a_corrupt_reading_on_either_controller(
    controller: FixedKLCoefficient | AdaptiveKLController,
    reading: float,
    fragment: str,
) -> None:
    with pytest.raises(BatchRefusal, match=fragment):
        controller.update(reading)
    assert controller.value == 0.04


@pytest.mark.parametrize("controller", [FixedKLCoefficient(), AdaptiveKLController()])
@pytest.mark.parametrize("reading", [None, object(), (0.05,)])
def test_update_refuses_a_reading_that_is_not_a_scalar(
    controller: FixedKLCoefficient | AdaptiveKLController,
    reading: Any,
) -> None:
    # The other arm of the same guard: a value that does not CONVERT is a
    # different failure from one that converts to something corrupt, and it
    # carries its own message. Not asserted here, because it is the settled
    # package-wide idiom rather than a property of this module: `float(raw)`
    # coerces, so a numeric STRING is accepted as a reading.
    with pytest.raises(BatchRefusal, match="does not convert to a scalar float"):
        controller.update(reading)
    assert controller.value == 0.04


@pytest.mark.parametrize("controller", [FixedKLCoefficient(), AdaptiveKLController()])
def test_update_refuses_a_bool_reading(
    controller: FixedKLCoefficient | AdaptiveKLController,
) -> None:
    # True is an int in Python; a KL controller still refuses to absorb it.
    with pytest.raises(BatchRefusal, match="is a bool"):
        controller.update(True)


def test_reading_above_the_dead_band_grows_the_coefficient_by_exactly_growth() -> None:
    controller = AdaptiveKLController()
    grown = controller.update(0.05)  # 0.05 > 0.02 * 1.5
    assert grown is not controller
    assert grown.value == 0.08  # 0.04 * 2.0 exactly
    assert isinstance(grown, AdaptiveKLController)
    # replace() must carry the rest of the configuration across unchanged.
    assert grown.target_kl == controller.target_kl
    assert grown.tolerance_band == controller.tolerance_band
    assert grown.growth == controller.growth
    assert grown.decay == controller.decay
    assert grown.minimum == controller.minimum
    assert grown.maximum is None
    assert controller.value == 0.04


def test_reading_below_the_dead_band_decays_the_coefficient_by_exactly_decay() -> None:
    controller = AdaptiveKLController()
    decayed = controller.update(0.01)  # 0.01 < 0.02 / 1.5
    assert decayed is not controller
    assert decayed.value == 0.02  # 0.04 * 0.5 exactly
    assert controller.value == 0.04


def test_reading_inside_the_dead_band_returns_the_same_object() -> None:
    # Identity, not equality: the source claims an unchanged reading returns
    # self, and a mutation that rebuilds an equal controller must fail this.
    controller = AdaptiveKLController()
    assert controller.update(0.02) is controller  # the target itself
    assert controller.update(0.025) is controller
    assert controller.update(0.014) is controller
    assert controller.value == 0.04


def test_upper_dead_band_edge_is_strict() -> None:
    # Exactly AT the upper edge nothing happens: a `>` flipped to `>=` would
    # grow here, and only a probe on the edge itself sees that off-by-one.
    controller = AdaptiveKLController()
    edge = controller.target_kl * controller.tolerance_band
    assert controller.update(edge) is controller
    grown = controller.update(edge + 1e-9)
    assert grown is not controller
    assert grown.value == 0.08


def test_lower_dead_band_edge_is_strict() -> None:
    controller = AdaptiveKLController()
    edge = controller.target_kl / controller.tolerance_band
    assert controller.update(edge) is controller
    decayed = controller.update(edge * (1.0 - 1e-9))
    assert decayed is not controller
    assert decayed.value == 0.02


def test_decay_past_minimum_clamps_to_a_new_controller_at_the_floor() -> None:
    controller = AdaptiveKLController(value=1.5e-8)
    decayed = controller.update(0.0)  # 1.5e-8 * 0.5 clamps up to 1e-8
    assert decayed is not controller
    assert decayed.value == 1.0e-8
    assert decayed.value == controller.minimum


def test_decay_at_minimum_clamps_back_to_self() -> None:
    # The clamp lands EXACTLY on the value in force: the controller must then
    # return itself, not an equal rebuild at the same value.
    controller = AdaptiveKLController(value=1.0e-8)
    assert controller.minimum == 1.0e-8
    assert controller.update(0.0) is controller


def test_growth_past_maximum_clamps_to_a_new_controller_at_the_ceiling() -> None:
    controller = AdaptiveKLController(value=0.04, maximum=0.05)
    grown = controller.update(0.05)  # 0.04 * 2.0 clamps down to 0.05
    assert grown is not controller
    assert grown.value == 0.05
    assert grown.value == controller.maximum


def test_growth_at_maximum_clamps_back_to_self() -> None:
    controller = AdaptiveKLController(value=0.05, maximum=0.05)
    assert controller.maximum is not None
    assert controller.update(0.05) is controller


def test_no_upper_clamp_when_maximum_is_none() -> None:
    controller = AdaptiveKLController(value=4.0)
    assert controller.maximum is None
    once = controller.update(1.0)
    twice = once.update(1.0)
    assert once.value == 8.0
    assert twice.value == 16.0
    assert controller.value == 4.0


def test_update_never_mutates_the_controller_it_was_called_on() -> None:
    controller = AdaptiveKLController()
    grown = controller.update(0.05)
    decayed = controller.update(0.001)
    assert grown is not decayed
    assert grown.value == 0.08
    assert decayed.value == 0.02
    assert controller.value == 0.04
    assert controller.coefficient() == 0.04


@pytest.mark.parametrize("controller", [FixedKLCoefficient(), AdaptiveKLController()])
def test_frozen_controllers_refuse_direct_state_assignment(
    controller: FixedKLCoefficient | AdaptiveKLController,
) -> None:
    with pytest.raises(FrozenInstanceError):
        controller.value = 0.5


def test_both_concrete_controllers_satisfy_the_protocol() -> None:
    assert isinstance(FixedKLCoefficient(), KLCoefficientController)
    assert isinstance(AdaptiveKLController(), KLCoefficientController)


def test_runtime_checkable_protocol_checks_method_presence_only() -> None:
    # @runtime_checkable asks only that the METHOD NAMES exist; it never
    # inspects signatures or return shapes, so _WrongShapeController passes
    # isinstance and would then fail at the call site. Both halves are pinned:
    # the structural acceptance, and the construction still refusing the
    # wrong-shaped coefficient reading, naming what came back.
    wrong = _WrongShapeController()
    assert isinstance(wrong, KLCoefficientController)
    with pytest.raises(LossConfigRefusal, match="returned '0.5'"):
        KLPenaltyLoss(controller=wrong)


def test_a_zero_coefficient_controller_is_refused_at_construction() -> None:
    assert isinstance(_ZeroCoefficientController(), KLCoefficientController)
    with pytest.raises(LossConfigRefusal, match="returned 0.0"):
        KLPenaltyLoss(controller=_ZeroCoefficientController())


def test_a_plain_float_is_not_a_controller() -> None:
    with pytest.raises(LossConfigRefusal, match="not float"):
        KLPenaltyLoss(controller=0.04)


def test_an_adaptive_controller_is_accepted_as_the_coefficient_source() -> None:
    loss_fn = KLPenaltyLoss(controller=AdaptiveKLController(value=0.09))
    assert loss_fn.controller.coefficient() == 0.09


@pytest.mark.parametrize(
    "field_name",
    ["mask_column", "reference_logprob_column", "component_name", "metric_name"],
)
@pytest.mark.parametrize("bad", ["", 7])
def test_every_name_field_of_kl_penalty_refuses_a_non_name(field_name: str, bad: object) -> None:
    # Parametrised over the WHOLE field list: a name field added later without
    # a guard is exactly the case a sampled test would not catch.
    with pytest.raises(LossConfigRefusal) as exc_info:
        KLPenaltyLoss(**{field_name: bad})
    assert field_name in str(exc_info.value)


@pytest.mark.parametrize(
    ("ceiling", "fragment"),
    [
        (0.0, "metric_ceiling=0.0"),
        (-2.0, "metric_ceiling=-2.0"),
        (True, "metric_ceiling=True"),
        (float("inf"), "metric_ceiling=inf"),
        (float("nan"), "metric_ceiling=nan"),
    ],
)
def test_metric_ceiling_must_be_finite_and_strictly_positive(
    ceiling: object, fragment: str
) -> None:
    # A non-finite bound makes every gate comparison against it silently True;
    # both infinite and NaN ceilings must be refused at construction, and so
    # must a non-positive one and the bool that is an int.
    with pytest.raises(LossConfigRefusal, match=fragment):
        KLPenaltyLoss(metric_ceiling=ceiling)


def test_declaration_carries_a_finite_ceiling_equal_to_the_field() -> None:
    expectation = KLPenaltyLoss(metric_ceiling=3.5).declaration().metrics[0]
    assert expectation.name == "kl_estimate"
    assert expectation.low == 0.0
    assert expectation.high == 3.5
    assert isinstance(expectation.high, float)
    assert math.isfinite(expectation.high)
    # 0.0 is the correct on-policy first-epoch reading, not a degenerate one.
    assert expectation.degenerate == ()
    assert KLPenaltyLoss().declaration().metrics[0].high == 10.0


def test_required_columns_and_declaration_follow_the_configured_names() -> None:
    loss_fn = KLPenaltyLoss(
        mask_column="m",
        reference_logprob_column="ref",
        component_name="kl_term",
        metric_name="k3_reading",
    )
    assert loss_fn.required_columns == ("m", "ref")
    declaration = loss_fn.declaration()
    assert declaration.components == ("kl_term",)
    assert declaration.metrics[0].name == "k3_reading"
    assert KLPenaltyLoss().required_columns == ("loss_mask", "reference_logprobs")


def test_declaration_and_observed_report_cannot_diverge() -> None:
    loss_fn = KLPenaltyLoss()
    out = loss_fn(_forward(_CURRENT), _batch())
    declared_components = set(loss_fn.declaration().components)
    observed_components = {component.name for component in out.components}
    assert observed_components == declared_components == {"kl_penalty"}
    declared_metrics = [metric.name for metric in loss_fn.declaration().metrics]
    assert [metric.name for metric in out.metrics] == declared_metrics == ["kl_estimate"]


def test_k3_estimate_is_the_supervised_token_mean_computed_by_hand() -> None:
    # k3(d) = expm1(d) - d with d = reference - current, per supervised token:
    #   row 0: d =  0.0-(-0.1) =  0.1 -> k3 = 0.105170918075648 - 0.1 = 0.005170918075648
    #   row 0: d = -0.5-(-0.3) = -0.2 -> k3 = -0.181269246922018 + 0.2 = 0.018730753077982
    #   row 0: d = -1.0-(-1.3) =  0.3 -> k3 = 0.349858807576003 - 0.3 = 0.049858807576003
    #   row 1 position 0 is masked OUT: its d = 9.0 would give k3 ~= 8102, so
    #   any leak of that token into the mean explodes the expectation below.
    #   row 1: d = -0.25-(-0.25) = 0.0 -> k3 = 0.0 exactly.
    # supervised = 4 tokens across 2 rows; mean = 0.073760478729633 / 4.
    controller = FixedKLCoefficient(value=0.5)
    out = KLPenaltyLoss(controller=controller)(_forward(_CURRENT), _batch())
    assert out.metrics[0].value == pytest.approx(0.018440119682408)
    assert out.loss == pytest.approx(0.009220059841204)  # 0.5 * the mean
    assert len(out.components) == 1
    component = out.components[0]
    assert component.name == "kl_penalty"
    assert component.observed is True
    assert component.weight == 0.5
    assert component.weight == controller.coefficient()
    assert component.contribution == pytest.approx(out.loss)
    assert len(out.metrics) == 1
    assert out.metrics[0].name == "kl_estimate"


def test_component_weight_is_the_controller_coefficient_at_call_time() -> None:
    controller = AdaptiveKLController().update(0.05)
    assert controller.coefficient() == 0.08
    out = KLPenaltyLoss(controller=controller)(_forward(_CURRENT), _batch())
    assert out.components[0].weight == 0.08
    assert out.loss == pytest.approx(0.001475209574593)  # 0.08 * the mean
    # The metric stays UNWEIGHTED: it is the reading the controller consumes.
    assert out.metrics[0].value == pytest.approx(0.018440119682408)


def test_on_policy_batch_reports_exactly_zero_and_zero_is_not_degenerate() -> None:
    # current == reference on every supervised token, so every k3 is exactly
    # 0.0. That reading is CORRECT on the first inner epoch, not evidence of a
    # broken estimator: the declaration must not list it as degenerate.
    batch = _batch(masks=((1, 1),), references=((-0.4, -0.7),))
    out = KLPenaltyLoss(controller=FixedKLCoefficient(value=0.3))(_forward([(-0.4, -0.7)]), batch)
    assert out.metrics[0] == MetricObservation(name="kl_estimate", value=0.0)
    assert out.metrics[0].value == 0.0
    assert out.loss == 0.0
    assert out.components[0].contribution == 0.0
    expectation = KLPenaltyLoss().declaration().metrics[0]
    assert expectation.degenerate == ()
    assert out.metrics[0].value not in expectation.degenerate


def test_microscopic_negative_k3_artefact_is_floored_to_zero() -> None:
    # d = 0.0 - (-1e-16) = 1e-16; the true k3 is d**2 / 2 = 5e-33. A correctly
    # rounded expm1 returns d exactly (k3 == 0.0); a libm one ulp low hands
    # back roughly -1e-32, inside (-1e-15, 0.0), and the floor must map it to
    # 0.0 rather than refusing. Either way the observable reading is 0.0.
    batch = _batch(masks=((1,),), references=((0.0,),))
    out = KLPenaltyLoss()(_forward([(-1e-16,)]), batch)
    assert out.metrics[0].value == 0.0
    assert out.metrics[0].value >= 0.0
    assert out.loss == 0.0


def test_an_unrepresentable_expm1_is_refused_not_clamped() -> None:
    batch = _batch(masks=((1,),), references=((800.0,),))
    with pytest.raises(BatchRefusal, match=r"exp\(800\.0\) is not representable"):
        KLPenaltyLoss()(_forward([(0.0,)]), batch)


def test_a_non_finite_k3_reading_is_refused_naming_the_value() -> None:
    # d = 1e308 - (-1e308) overflows to inf; expm1(inf) - inf is nan.
    batch = _batch(masks=((1,),), references=((1e308,),))
    with pytest.raises(BatchRefusal, match="k3 value nan"):
        KLPenaltyLoss()(_forward([(-1e308,)]), batch)


def test_call_time_revalidation_refuses_a_controller_whose_reading_decayed() -> None:
    # Valid 0.5 at construction, decayed to 0.0 by the time the loss prices
    # with it: the re-check inside __call__ is what refuses, not post_init.
    loss_fn = KLPenaltyLoss(controller=_DecayedController())
    with pytest.raises(LossConfigRefusal, match="returned 0.0"):
        loss_fn(_forward(_CURRENT), _batch())


def test_missing_reference_column_refused() -> None:
    batch = ExperienceBatch(columns={"loss_mask": _MASKS})
    with pytest.raises(BatchRefusal) as exc_info:
        KLPenaltyLoss()(_forward(_CURRENT), batch)
    message = str(exc_info.value)
    assert "reference_logprobs" in message
    assert "1 of 2" in message


def test_empty_batch_refused() -> None:
    batch = _batch(masks=(), references=())
    with pytest.raises(BatchRefusal) as exc_info:
        KLPenaltyLoss()(_forward([]), batch)
    message = str(exc_info.value)
    assert "0 of at least 1 required batch rows" in message
    assert "never 0.0" in message


def test_zero_supervised_tokens_is_refused_never_zero() -> None:
    batch = _batch(masks=((0, 0), (0,)), references=((-0.1, -0.2), (-0.3,)))
    rows = [(-0.15, -0.25), (-0.35,)]
    with pytest.raises(SupervisionRefusal) as exc_info:
        KLPenaltyLoss()(_forward(rows), batch)
    message = str(exc_info.value)
    assert "0 supervised tokens" in message
    assert "2 offered rows" in message


def test_wrong_forward_row_count_refused() -> None:
    with pytest.raises(BatchRefusal) as exc_info:
        KLPenaltyLoss()(_forward([_CURRENT[0]]), _batch())
    message = str(exc_info.value)
    assert "forward_fn returned 1 rows" in message
    assert "batch of 2" in message


def test_non_finite_reference_reading_refused() -> None:
    batch = _batch(masks=((1, 1),), references=((0.25, float("nan")),))
    with pytest.raises(BatchRefusal) as exc_info:
        KLPenaltyLoss()(_forward([(-0.1, -0.2)]), batch)
    message = str(exc_info.value)
    assert "reference_logprobs" in message
    assert "row 0, position 1" in message
    assert "nan" in message


def test_current_reading_that_does_not_convert_refused() -> None:
    batch = _batch(masks=((1,),), references=((0.0,),))
    with pytest.raises(BatchRefusal) as exc_info:
        KLPenaltyLoss()(_forward([(object(),)]), batch)
    message = str(exc_info.value)
    assert "current_logprobs" in message
    assert "does not convert" in message
    assert "row 0, position 0" in message


def test_fractional_mask_entry_refused() -> None:
    batch = _batch(masks=((0.5,),), references=((0.0,),))
    with pytest.raises(BatchRefusal, match=r"mask entry at row 0, position 0 is 0\.5"):
        KLPenaltyLoss()(_forward([(0.1,)]), batch)


def test_reference_row_that_is_not_iterable_refused() -> None:
    batch = _batch(masks=((1,),), references=(7,))
    with pytest.raises(BatchRefusal) as exc_info:
        KLPenaltyLoss()(_forward([(0.1,)]), batch)
    message = str(exc_info.value)
    assert "reference_logprobs row 0" in message
    assert "not iterable" in message
    assert "int" in message


def test_reference_row_longer_than_the_mask_refused() -> None:
    batch = _batch(masks=((1, 1),), references=((0.0, -0.1, -0.2),))
    with pytest.raises(BatchRefusal, match=r"lengths are \(2, 3, 2\)"):
        KLPenaltyLoss()(_forward([(-0.05, -0.15)]), batch)


def test_reference_row_shorter_than_the_mask_refused() -> None:
    batch = _batch(masks=((1, 1),), references=((0.0,),))
    with pytest.raises(BatchRefusal, match=r"lengths are \(2, 1, 2\)"):
        KLPenaltyLoss()(_forward([(-0.05, -0.15)]), batch)


def test_current_row_shorter_than_the_mask_refused() -> None:
    # A distinct length disagreement of its own: a mutation that stopped
    # comparing the current row against the others fails only this leg.
    batch = _batch(masks=((1, 1),), references=((0.0, -0.1),))
    with pytest.raises(BatchRefusal, match=r"lengths are \(2, 2, 1\)"):
        KLPenaltyLoss()(_forward([(-0.05,)]), batch)
