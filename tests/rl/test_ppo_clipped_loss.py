"""Tests for the PPO clipped surrogate policy loss and the value function loss."""

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
    PPOClippedPolicyLoss,
    SupervisionRefusal,
    ValueFunctionLoss,
)
from foundationscale.rl.interfaces import LossOutput

# Healthy two-row batches: ratios are identically 1 (current == old == 0.0), so
# the clip is inert and the default advantages average to 1.0.
_GOOD_MASKS = ((1, 1), (1, 1))
_GOOD_OLD_LOGPROBS = ((0.0, 0.0), (0.0, 0.0))
_GOOD_ADVANTAGES = ((1.0, 1.0), (1.0, 1.0))
_GOOD_RETURNS = ((1.0, 1.0), (1.0, 1.0))

_POLICY_NAME_FIELDS = (
    "mask_column",
    "old_logprob_column",
    "advantage_column",
    "component_name",
    "metric_name",
)
_VALUE_NAME_FIELDS = (
    "mask_column",
    "return_column",
    "old_value_column",
    "component_name",
    "metric_name",
)


def _forward(rows: Any) -> ForwardFn:
    materialised = list(rows)
    return lambda _batch: list(materialised)


def _good_forward() -> ForwardFn:
    return _forward(((0.0, 0.0), (0.0, 0.0)))


def _policy_batch(
    masks: Any = _GOOD_MASKS,
    old_logprobs: Any = _GOOD_OLD_LOGPROBS,
    advantages: Any = _GOOD_ADVANTAGES,
    mask_column: str = "loss_mask",
    old_logprob_column: str = "old_logprobs",
    advantage_column: str = "advantages",
) -> ExperienceBatch:
    return ExperienceBatch(
        columns={
            mask_column: masks,
            old_logprob_column: old_logprobs,
            advantage_column: advantages,
        }
    )


def _value_batch(
    masks: Any = _GOOD_MASKS,
    returns: Any = _GOOD_RETURNS,
    old_values: Any = None,
) -> ExperienceBatch:
    columns: dict[str, Any] = {"loss_mask": masks, "returns": returns}
    if old_values is not None:
        columns["old_values"] = old_values
    return ExperienceBatch(columns=columns)


def _policy_token(
    log_ratio: Any,
    advantage: Any,
    loss_fn: PPOClippedPolicyLoss | None = None,
) -> LossOutput:
    """Price one old-logprob-0.0 supervised token at the given log-ratio."""
    resolved = PPOClippedPolicyLoss() if loss_fn is None else loss_fn
    batch = _policy_batch(
        masks=((1,),),
        old_logprobs=((0.0,),),
        advantages=((advantage,),),
    )
    return resolved(_forward([[log_ratio]]), batch)


class TestPPOClippedPolicyLossContract:
    def test_healthy_batch_reports_one_component_and_the_diagnostic(self) -> None:
        out = PPOClippedPolicyLoss()(_good_forward(), _policy_batch())
        assert out.loss == pytest.approx(-1.0)
        component = out.components[0]
        assert component.name == "ppo_policy_loss"
        assert component.weight == 1.0
        assert component.observed is True
        assert component.contribution == out.loss
        assert out.metrics == (MetricObservation(name="clip_fraction", value=0.0),)

    def test_required_columns_are_stable(self) -> None:
        assert PPOClippedPolicyLoss().required_columns == (
            "loss_mask",
            "old_logprobs",
            "advantages",
        )

    def test_declaration_bounds_and_empty_degenerate(self) -> None:
        declaration = PPOClippedPolicyLoss().declaration()
        assert declaration.components == ("ppo_policy_loss",)
        expectation = declaration.metrics[0]
        assert expectation.name == "clip_fraction"
        assert expectation.low == 0.0
        assert expectation.high == 1.0
        assert expectation.degenerate == ()

    def test_custom_names_propagate_to_batch_components_and_metrics(self) -> None:
        loss_fn = PPOClippedPolicyLoss(
            mask_column="m",
            old_logprob_column="o",
            advantage_column="a",
            component_name="pg",
            metric_name="cf",
        )
        assert loss_fn.required_columns == ("m", "o", "a")
        batch = _policy_batch(
            masks=((1,),),
            old_logprobs=((0.0,),),
            advantages=((2.0,),),
            mask_column="m",
            old_logprob_column="o",
            advantage_column="a",
        )
        out = loss_fn(_forward([[0.0]]), batch)
        assert out.loss == pytest.approx(-2.0)
        assert [component.name for component in out.components] == ["pg"]
        assert [metric.name for metric in out.metrics] == ["cf"]


class TestPPOClippedPolicyLossArithmetic:
    def test_ratio_inside_the_band_makes_the_clip_inert(self) -> None:
        # eps = 0.2 -> interval [0.8, 1.2]; ratio = exp(ln 1.1) ~ 1.1 is inside.
        # score = min(1.1 * 2.0, 1.1 * 2.0) = 2.2; loss = -(2.2 / 1) = -2.2.
        # This leg alone cannot see a dropped clip, which is why the other four
        # legs exist; it pins that an inside-band ratio passes through UNCHANGED.
        out = _policy_token(math.log(1.1), 2.0)
        assert out.loss == pytest.approx(-2.2)
        assert out.metrics == (MetricObservation(name="clip_fraction", value=0.0),)

    def test_above_high_positive_advantage_the_clip_bites(self) -> None:
        # ratio = exp(ln 1.5) ~ 1.5 > 1.2 with advantage +2.0.
        # unclipped = 1.5 * 2.0 = 3.0; clipped = 1.2 * 2.0 = 2.4.
        # min picks the CLIPPED 2.4; an unclipped implementation reports -3.0.
        out = _policy_token(math.log(1.5), 2.0)
        assert out.loss == pytest.approx(-2.4)
        assert out.metrics == (MetricObservation(name="clip_fraction", value=1.0),)

    def test_above_high_negative_advantage_the_clip_does_not_bite(self) -> None:
        # The backwards-calculation trap: ratio ~ 1.5 > 1.2 with advantage -2.0.
        # unclipped = -3.0; clipped = 1.2 * -2.0 = -2.4.
        # min is over PESSIMISM: it picks the UNCLIPPED -3.0, so loss = +3.0.
        # An implementation that always uses the clipped term reports +2.4.
        out = _policy_token(math.log(1.5), -2.0)
        assert out.loss == pytest.approx(3.0)
        assert out.loss != pytest.approx(2.4)
        assert out.metrics == (MetricObservation(name="clip_fraction", value=1.0),)

    def test_below_low_positive_advantage_keeps_the_unclipped_term(self) -> None:
        # ratio = exp(ln 0.5) = 0.5 < 0.8 with advantage +2.0.
        # unclipped = 1.0; clipped = 0.8 * 2.0 = 1.6; min picks unclipped 1.0,
        # so below the band the clip only bites for a NEGATIVE advantage.
        out = _policy_token(math.log(0.5), 2.0)
        assert out.loss == pytest.approx(-1.0)
        assert out.metrics == (MetricObservation(name="clip_fraction", value=1.0),)

    def test_below_low_negative_advantage_the_clip_bites(self) -> None:
        # ratio = 0.5 < 0.8 with advantage -2.0:
        # unclipped = -1.0; clipped = 0.8 * -2.0 = -1.6;
        # min picks the CLIPPED -1.6, so loss = +1.6.
        out = _policy_token(math.log(0.5), -2.0)
        assert out.loss == pytest.approx(1.6)
        assert out.metrics == (MetricObservation(name="clip_fraction", value=1.0),)

    def test_weight_scales_the_reported_component(self) -> None:
        # Same inert term as the inside-band leg (2.2), weight 2.0 -> -4.4; a
        # mutation dropping the weight factor is caught here.
        loss_fn = PPOClippedPolicyLoss(weight=2.0)
        out = _policy_token(math.log(1.1), 2.0, loss_fn)
        assert out.loss == pytest.approx(-4.4)
        assert out.components[0].weight == 2.0
        assert out.components[0].contribution == out.loss

    def test_asymmetric_band_clips_on_a_different_side_than_the_default(self) -> None:
        # eps_low = 0.05, eps_high = 0.5 -> interval [0.95, 1.5].
        # token 0: ratio ~ 1.3, advantage +2.0 -- INSIDE the asymmetric band
        # (inert term 1.3 * 2.0 = 2.6) but OUTSIDE the symmetric [0.8, 1.2]
        # high side (clipped term 1.2 * 2.0 = 2.4).
        # token 1: ratio ~ 0.9, advantage -2.0 -- INSIDE the symmetric band
        # (inert term 0.9 * -2.0 = -1.8) but OUTSIDE the asymmetric LOW side:
        # min(0.9 * -2.0, 0.95 * -2.0) = min(-1.8, -1.9) = -1.9.
        # asymmetric: -(2.6 - 1.9) / 2 = -0.35; symmetric: -(2.4 - 1.8) / 2 = -0.3.
        asymmetric = PPOClippedPolicyLoss(clip_epsilon_low=0.05, clip_epsilon_high=0.5)
        batch = _policy_batch(
            masks=((1, 1),),
            old_logprobs=((0.0, 0.0),),
            advantages=((2.0, -2.0),),
        )
        forward = _forward([[math.log(1.3), math.log(0.9)]])
        asym_out = asymmetric(forward, batch)
        sym_out = PPOClippedPolicyLoss()(forward, batch)
        assert asym_out.loss == pytest.approx(-0.35)
        assert sym_out.loss == pytest.approx(-0.3)
        assert asym_out.loss != pytest.approx(sym_out.loss)

    def test_clip_fraction_counts_exactly_the_out_of_band_tokens(self) -> None:
        # ratios 1.5, 0.5, 1.1, 1.0 against [0.8, 1.2]: tokens 0 and 1 clip.
        # terms: min(1.5, 1.2) = 1.2; min(0.5, 0.8) = 0.5; 1.1 and 1.0 inert.
        # loss = -(1.2 + 0.5 + 1.1 + 1.0) / 4 = -0.95; 2 of 4 clip -> 0.5. A
        # wrong denominator (e.g. batch positions) would report 0.4 instead.
        batch = _policy_batch(
            masks=((1, 1, 1, 1),),
            old_logprobs=((0.0, 0.0, 0.0, 0.0),),
            advantages=((1.0, 1.0, 1.0, 1.0),),
        )
        forward = _forward([[math.log(1.5), math.log(0.5), math.log(1.1), 0.0]])
        out = PPOClippedPolicyLoss()(forward, batch)
        assert out.loss == pytest.approx(-0.95)
        assert out.metrics == (MetricObservation(name="clip_fraction", value=0.5),)

    def test_unit_ratios_report_zero_clip_fraction_and_it_is_not_degenerate(self) -> None:
        # First-inner-epoch reality: current == old everywhere, ratio == 1.0,
        # so the clip fraction is 0.0 and that is CORRECT, not a broken clip --
        # the declaration must therefore NOT name 0.0 as degenerate.
        batch = _policy_batch(
            masks=((1, 1),),
            old_logprobs=((-0.5, 0.25),),
            advantages=((2.0, 3.0),),
        )
        out = PPOClippedPolicyLoss()(_forward([[-0.5, 0.25]]), batch)
        # inert terms 2.0 and 3.0 -> loss = -(5.0 / 2) = -2.5
        assert out.loss == pytest.approx(-2.5)
        assert out.metrics == (MetricObservation(name="clip_fraction", value=0.0),)
        assert PPOClippedPolicyLoss().declaration().metrics[0].degenerate == ()

    def test_supervised_denominator_spans_rows_and_ignores_masked_tokens(self) -> None:
        # Two rows, three supervised tokens; the masked positions carry NaNs that
        # would blow up if the denominator ever scalarised them.
        # supervised terms: 1 * 2.0 = 2.0; 1 * 3.0 = 3.0; 1 * -4.0 = -4.0.
        # loss = -(2.0 + 3.0 - 4.0) / 3 = -1/3. Dividing by 5 batch positions
        # instead of 3 supervised tokens would report -0.2.
        nan = float("nan")
        batch = _policy_batch(
            masks=((1, 0, 1), (0, 1)),
            old_logprobs=((0.0, nan, 0.0), (nan, 0.0)),
            advantages=((2.0, nan, 3.0), (nan, -4.0)),
        )
        out = PPOClippedPolicyLoss()(_forward([[0.0, 0.0, 0.0], [0.0, 0.0]]), batch)
        assert out.loss == pytest.approx(-1 / 3)
        assert out.components[0].contribution == out.loss

    def test_boolean_mask_entries_are_accepted(self) -> None:
        # bool mask entries are honest 0/1 supervision (unlike bool ADVANTAGES,
        # which are refused below); only the True position counts.
        batch = _policy_batch(
            masks=((True, False),),
            old_logprobs=((0.0, 0.0),),
            advantages=((4.0, 5.0),),
        )
        out = PPOClippedPolicyLoss()(_forward([[0.0, 0.0]]), batch)
        assert out.loss == pytest.approx(-4.0)


class TestPPOClippedPolicyLossConfig:
    @pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
    def test_clip_epsilon_refused(self, bad: float) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            PPOClippedPolicyLoss(clip_epsilon=bad)
        assert f"clip_epsilon={bad!r}" in str(exc_info.value)

    def test_clip_epsilon_bool_is_not_a_width(self) -> None:
        # Python's True is an int, so it would pass a plain numeric check; the
        # source deliberately refuses it and this pin makes that refusal stick.
        with pytest.raises(LossConfigRefusal) as exc_info:
            PPOClippedPolicyLoss(clip_epsilon=True)
        assert "clip_epsilon=True" in str(exc_info.value)

    def test_clip_epsilon_at_one_collapses_the_lower_bound(self) -> None:
        # The upper end of the width: 1.0 - 1.0 = 0.0 is not a positive interval.
        with pytest.raises(LossConfigRefusal) as exc_info:
            PPOClippedPolicyLoss(clip_epsilon=1.0)
        message = str(exc_info.value)
        assert "resolved clip interval" in message
        assert "must stay positive" in message

    def test_low_override_at_or_above_one_collapses_the_lower_bound(self) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            PPOClippedPolicyLoss(clip_epsilon_low=1.5)
        message = str(exc_info.value)
        assert "1.5" in message
        assert "must stay positive" in message

    @pytest.mark.parametrize("field", ["clip_epsilon_low", "clip_epsilon_high"])
    @pytest.mark.parametrize("bad", [0.0, -0.25, float("nan")])
    def test_asymmetric_width_refused(self, field: str, bad: float) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            PPOClippedPolicyLoss(**{field: bad})
        assert f"{field}={bad!r}" in str(exc_info.value)

    @pytest.mark.parametrize("field", ["clip_epsilon_low", "clip_epsilon_high"])
    def test_asymmetric_width_bool_refused(self, field: str) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            PPOClippedPolicyLoss(**{field: True})
        assert f"{field}=True" in str(exc_info.value)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_weight_refused(self, bad: float) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            PPOClippedPolicyLoss(weight=bad)
        assert f"weight={bad!r}" in str(exc_info.value)

    def test_weight_bool_refused(self) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            PPOClippedPolicyLoss(weight=True)
        assert "weight=True" in str(exc_info.value)

    @pytest.mark.parametrize("field", _POLICY_NAME_FIELDS)
    @pytest.mark.parametrize("bad", ["", 7])
    def test_every_name_field_refuses_a_non_name(self, field: str, bad: object) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            PPOClippedPolicyLoss(**{field: bad})
        assert field in str(exc_info.value)


class TestPPOClippedPolicyLossBatchRefusals:
    def test_missing_advantage_column_refused(self) -> None:
        batch = ExperienceBatch(
            columns={"loss_mask": _GOOD_MASKS, "old_logprobs": _GOOD_OLD_LOGPROBS}
        )
        with pytest.raises(BatchRefusal) as exc_info:
            PPOClippedPolicyLoss()(_good_forward(), batch)
        message = str(exc_info.value)
        assert "1 of 3" in message
        assert "{advantages}" in message

    def test_empty_batch_refused(self) -> None:
        batch = _policy_batch((), (), ())
        with pytest.raises(BatchRefusal) as exc_info:
            PPOClippedPolicyLoss()(_forward([]), batch)
        message = str(exc_info.value)
        assert "0 of at least 1" in message
        assert "ppo policy loss" in message

    def test_forward_row_count_mismatch_refused(self) -> None:
        with pytest.raises(BatchRefusal) as exc_info:
            PPOClippedPolicyLoss()(_forward([[0.0, 0.0]]), _policy_batch())
        message = str(exc_info.value)
        assert "returned 1" in message
        assert "of 2" in message

    def test_ragged_token_row_refused(self) -> None:
        batch = _policy_batch(
            masks=((1, 1),),
            old_logprobs=((0.0,),),
            advantages=((1.0, 1.0),),
        )
        with pytest.raises(BatchRefusal) as exc_info:
            PPOClippedPolicyLoss()(_forward([[0.0, 0.0]]), batch)
        message = str(exc_info.value)
        assert "row 0" in message
        assert "mask/old/advantage/current" in message
        assert "(2, 1, 2, 2)" in message

    def test_non_iterable_column_refused_at_batch_construction(self) -> None:
        # _outer_rows guards this shape, but a non-iterable COLUMN never
        # reaches it: ExperienceBatch refuses at construction, because a
        # column must be a sized sequence before any loss sees it. Pinning
        # the real boundary rather than the one the loss would like to own.
        with pytest.raises(BatchRefusal) as exc_info:
            _policy_batch(old_logprobs=7)
        message = str(exc_info.value)
        assert "old_logprobs" in message
        assert "has no row count" in message

    def test_non_iterable_forward_output_refused(self) -> None:
        # The reachable arm of the same guard: forward_fn's return value is
        # NOT a column, so nothing upstream has checked it for a row count.
        def forward_fn(_batch: ExperienceBatch) -> Any:
            return 7

        with pytest.raises(BatchRefusal) as exc_info:
            PPOClippedPolicyLoss()(forward_fn, _policy_batch())
        message = str(exc_info.value)
        assert "forward_fn(batch)" in message
        assert "was not iterable" in message

    def test_non_iterable_mask_row_refused(self) -> None:
        batch = _policy_batch(masks=(7, (1, 1)))
        with pytest.raises(BatchRefusal) as exc_info:
            PPOClippedPolicyLoss()(_good_forward(), batch)
        message = str(exc_info.value)
        assert "loss_mask" in message
        assert "row 0" in message

    def test_fractional_mask_entry_refused(self) -> None:
        batch = _policy_batch(masks=((0.5, 1), (1, 1)))
        with pytest.raises(BatchRefusal) as exc_info:
            PPOClippedPolicyLoss()(_good_forward(), batch)
        message = str(exc_info.value)
        assert "0.5" in message
        assert "0 or 1" in message

    def test_unconvertible_mask_entry_refused(self) -> None:
        batch = _policy_batch(masks=((1, object()), (1, 1)))
        with pytest.raises(BatchRefusal) as exc_info:
            PPOClippedPolicyLoss()(_good_forward(), batch)
        message = str(exc_info.value)
        assert "row 0" in message
        assert "position 1" in message
        assert "0 or 1" in message

    def test_boolean_advantage_refused(self) -> None:
        with pytest.raises(BatchRefusal) as exc_info:
            _policy_token(0.0, True)
        assert "a bool is not a measured token advantage" in str(exc_info.value)

    def test_non_finite_advantage_refused(self) -> None:
        with pytest.raises(BatchRefusal) as exc_info:
            _policy_token(0.0, float("nan"))
        message = str(exc_info.value)
        assert "nan" in message
        assert "not finite" in message

    def test_unconvertible_advantage_refused(self) -> None:
        with pytest.raises(BatchRefusal) as exc_info:
            _policy_token(0.0, object())
        assert "does not convert" in str(exc_info.value)

    def test_non_finite_old_logprob_refused(self) -> None:
        batch = _policy_batch(
            masks=((1, 1),),
            old_logprobs=((float("nan"), 0.0),),
            advantages=((1.0, 1.0),),
        )
        with pytest.raises(BatchRefusal) as exc_info:
            PPOClippedPolicyLoss()(_forward([[0.0, 0.0]]), batch)
        message = str(exc_info.value)
        assert "old_logprobs" in message
        assert "nan" in message

    def test_unconvertible_current_logprob_refused(self) -> None:
        with pytest.raises(BatchRefusal) as exc_info:
            _policy_token(object(), 1.0)
        message = str(exc_info.value)
        assert "current_logprobs" in message
        assert "does not convert" in message

    def test_zero_supervised_tokens_refused_never_zero(self) -> None:
        batch = _policy_batch(masks=((0, 0), (0, 0)))
        with pytest.raises(SupervisionRefusal) as exc_info:
            PPOClippedPolicyLoss()(_good_forward(), batch)
        message = str(exc_info.value)
        assert "0 supervised tokens" in message
        assert "2 offered rows" in message

    def test_unrepresentable_exp_refused_rather_than_clamped(self) -> None:
        # exp(1000.0) overflows Python's float: the source states it REFUSES an
        # unrepresentable ratio rather than clamping it, and this pins that.
        with pytest.raises(BatchRefusal) as exc_info:
            _policy_token(1000.0, 1.0)
        message = str(exc_info.value)
        assert "exp(1000.0)" in message
        assert "not representable" in message

    def test_underflowed_ratio_refused_rather_than_zero(self) -> None:
        # exp(-1000.0) underflows to exactly 0.0, which is not positive: refused.
        with pytest.raises(BatchRefusal) as exc_info:
            _policy_token(-1000.0, 1.0)
        message = str(exc_info.value)
        assert "0.0" in message
        assert "positive finite" in message


class TestValueFunctionLossContract:
    def test_required_columns_omit_old_values_without_clipping(self) -> None:
        assert ValueFunctionLoss().required_columns == ("loss_mask", "returns")

    def test_required_columns_add_old_values_with_clipping(self) -> None:
        assert ValueFunctionLoss(clip_epsilon=0.2).required_columns == (
            "loss_mask",
            "returns",
            "old_values",
        )

    def test_declaration_omits_the_metric_without_clipping(self) -> None:
        declaration = ValueFunctionLoss().declaration()
        assert declaration.components == ("value_loss",)
        assert declaration.metrics == ()

    def test_declaration_declares_the_metric_with_clipping(self) -> None:
        declaration = ValueFunctionLoss(clip_epsilon=0.2).declaration()
        assert declaration.components == ("value_loss",)
        expectation = declaration.metrics[0]
        assert expectation.name == "value_clip_fraction"
        assert expectation.low == 0.0
        assert expectation.high == 1.0
        assert expectation.degenerate == ()


class TestValueFunctionLossArithmetic:
    def test_plain_mse_by_hand(self) -> None:
        # errors: (1.0 - 1.5)^2 = 0.25; (2.0 - 0.0)^2 = 4.0; (0.5 - 2.0)^2 =
        # 2.25; sum = 6.5 over 3 supervised tokens -> loss = 6.5 / 3. No
        # clipping is configured, so the metric channel is EMPTY -- unmeasured,
        # never reported as 0.0 -- and the declaration agrees.
        batch = _value_batch(
            masks=((1, 1, 1),),
            returns=((1.5, 0.0, 2.0),),
        )
        out = ValueFunctionLoss()(_forward([[1.0, 2.0, 0.5]]), batch)
        assert out.loss == pytest.approx(6.5 / 3)
        assert out.metrics == ()
        component = out.components[0]
        assert component.name == "value_loss"
        assert component.observed is True
        assert component.contribution == out.loss
        assert ValueFunctionLoss().declaration().metrics == ()

    def test_weight_scales_the_mse_component(self) -> None:
        # error (1.0 - 3.0)^2 = 4.0 over 1 token; weight 2.0 -> 8.0.
        loss_fn = ValueFunctionLoss(weight=2.0)
        batch = _value_batch(masks=((1,),), returns=((3.0,),))
        out = loss_fn(_forward([[1.0]]), batch)
        assert out.loss == pytest.approx(8.0)
        assert out.components[0].weight == 2.0
        assert out.components[0].contribution == out.loss

    def test_clip_max_picks_each_arm_once(self) -> None:
        # eps = 0.2, targets 0.0.
        # token 0: old = 1.0, prediction = -0.5 moved PAST the target.
        #   unclipped = (-0.5)^2 = 0.25; shift = clamp(-1.5, -0.2, 0.2) = -0.2;
        #   shifted = 0.8; clipped = 0.64 > 0.25 -> max picks CLIPPED, applied.
        # token 1: old = 0.0, prediction = 1.0.
        #   unclipped = 1.0; shift = 0.2; shifted = 0.2; clipped = 0.04 < 1.0
        #   -> max picks UNCLIPPED, not applied.
        # loss = (0.64 + 1.0) / 2 = 0.82; applied 1 of 2 -> fraction 0.5. A
        # swapped min would report (0.25 + 0.04) / 2 = 0.145 instead.
        batch = _value_batch(
            masks=((1, 1),),
            returns=((0.0, 0.0),),
            old_values=((1.0, 0.0),),
        )
        out = ValueFunctionLoss(clip_epsilon=0.2)(_forward([[-0.5, 1.0]]), batch)
        assert out.loss == pytest.approx(0.82)
        assert out.metrics == (MetricObservation(name="value_clip_fraction", value=0.5),)
        assert ValueFunctionLoss(clip_epsilon=0.2).declaration().metrics[0].degenerate == ()

    def test_equal_branches_count_as_not_applied(self) -> None:
        # prediction == old value, so shift = 0.0 and clipped == unclipped =
        # (1.0 - 2.0)^2 = 1.0. The strictly-greater count stays 0: a batch that
        # never used the clipped arm correctly reports 0.0, not breakage. A >=
        # mutation would report 1.0 here.
        batch = _value_batch(
            masks=((1,),),
            returns=((2.0,),),
            old_values=((1.0,),),
        )
        out = ValueFunctionLoss(clip_epsilon=0.2)(_forward([[1.0]]), batch)
        assert out.loss == pytest.approx(1.0)
        assert out.metrics == (MetricObservation(name="value_clip_fraction", value=0.0),)

    def test_supervised_denominator_spans_rows_and_ignores_masked_tokens(self) -> None:
        # masked positions carry NaNs in returns AND predictions; only the three
        # supervised positions may be scalarised.
        # errors: (1.0 - 2.0)^2 = 1.0; (0.0 - 0.0)^2 = 0.0; (0.0 - 4.0)^2 = 16.
        # loss = 17.0 / 3; a denominator of 5 batch positions would give 3.4.
        nan = float("nan")
        batch = _value_batch(
            masks=((1, 0), (1, 1)),
            returns=((2.0, nan), (0.0, 4.0)),
        )
        out = ValueFunctionLoss()(_forward([[1.0, nan], [0.0, 0.0]]), batch)
        assert out.loss == pytest.approx(17 / 3)
        assert out.metrics == ()


class TestValueFunctionLossConfig:
    @pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf")])
    def test_clip_epsilon_refused(self, bad: float) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            ValueFunctionLoss(clip_epsilon=bad)
        assert f"clip_epsilon={bad!r}" in str(exc_info.value)

    def test_clip_epsilon_bool_is_not_a_width(self) -> None:
        # True must not smuggle through the numeric check as 1.0.
        with pytest.raises(LossConfigRefusal) as exc_info:
            ValueFunctionLoss(clip_epsilon=True)
        assert "clip_epsilon=True" in str(exc_info.value)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_weight_refused(self, bad: float) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            ValueFunctionLoss(weight=bad)
        assert f"weight={bad!r}" in str(exc_info.value)

    def test_weight_bool_refused(self) -> None:
        with pytest.raises(LossConfigRefusal) as exc_info:
            ValueFunctionLoss(weight=True)
        assert "weight=True" in str(exc_info.value)

    @pytest.mark.parametrize("field", _VALUE_NAME_FIELDS)
    @pytest.mark.parametrize("bad", ["", 7])
    def test_every_name_field_refuses_a_non_name(self, field: str, bad: object) -> None:
        # Even metric_name is validated with clipping OFF: a dormant name is
        # still a name and must be a non-empty string.
        with pytest.raises(LossConfigRefusal) as exc_info:
            ValueFunctionLoss(**{field: bad})
        assert field in str(exc_info.value)


class TestValueFunctionLossBatchRefusals:
    def test_missing_return_column_refused_without_clipping(self) -> None:
        batch = ExperienceBatch(columns={"loss_mask": _GOOD_MASKS})
        with pytest.raises(BatchRefusal) as exc_info:
            ValueFunctionLoss()(_good_forward(), batch)
        message = str(exc_info.value)
        assert "1 of 2" in message
        assert "{returns}" in message

    def test_missing_old_value_column_refused_with_clipping(self) -> None:
        with pytest.raises(BatchRefusal) as exc_info:
            ValueFunctionLoss(clip_epsilon=0.2)(_good_forward(), _value_batch())
        message = str(exc_info.value)
        assert "1 of 3" in message
        assert "{old_values}" in message

    def test_empty_batch_refused(self) -> None:
        batch = _value_batch((), ())
        with pytest.raises(BatchRefusal) as exc_info:
            ValueFunctionLoss()(_forward([]), batch)
        message = str(exc_info.value)
        assert "0 of at least 1" in message
        assert "ppo value loss" in message

    def test_forward_row_count_mismatch_refused(self) -> None:
        with pytest.raises(BatchRefusal) as exc_info:
            ValueFunctionLoss()(_forward([[0.0, 0.0]]), _value_batch())
        message = str(exc_info.value)
        assert "returned 1" in message
        assert "of 2" in message

    def test_non_iterable_forward_output_refused(self) -> None:
        with pytest.raises(BatchRefusal) as exc_info:
            ValueFunctionLoss()(lambda _batch: 7, _value_batch())
        message = str(exc_info.value)
        assert "forward_fn" in message
        assert "not iterable" in message

    def test_ragged_row_without_clipping_refused(self) -> None:
        batch = _value_batch(masks=((1, 1),), returns=((1.0, 2.0, 3.0),))
        with pytest.raises(BatchRefusal) as exc_info:
            ValueFunctionLoss()(_forward([[0.0, 0.0]]), batch)
        message = str(exc_info.value)
        assert "row 0" in message
        assert "mask/returns/predictions" in message
        assert "(2, 3, 2)" in message

    def test_ragged_row_with_clipping_refused(self) -> None:
        batch = _value_batch(
            masks=((1, 1),),
            returns=((1.0, 2.0),),
            old_values=((1.0,),),
        )
        with pytest.raises(BatchRefusal) as exc_info:
            ValueFunctionLoss(clip_epsilon=0.2)(_forward([[0.0, 0.0]]), batch)
        message = str(exc_info.value)
        assert "row 0" in message
        assert "mask/returns/old-values/predictions" in message
        assert "(2, 2, 1, 2)" in message

    def test_non_finite_return_refused(self) -> None:
        batch = _value_batch(masks=((1, 1),), returns=((float("nan"), 1.0),))
        with pytest.raises(BatchRefusal) as exc_info:
            ValueFunctionLoss()(_forward([[0.0, 0.0]]), batch)
        message = str(exc_info.value)
        assert "returns" in message
        assert "nan" in message

    def test_non_finite_prediction_refused(self) -> None:
        batch = _value_batch(masks=((1,),), returns=((1.0,),))
        with pytest.raises(BatchRefusal) as exc_info:
            ValueFunctionLoss()(_forward([[float("inf")]]), batch)
        message = str(exc_info.value)
        assert "value predictions" in message
        assert "inf" in message

    def test_non_finite_old_value_refused_with_clipping(self) -> None:
        batch = _value_batch(
            masks=((1,),),
            returns=((1.0,),),
            old_values=((float("nan"),),),
        )
        with pytest.raises(BatchRefusal) as exc_info:
            ValueFunctionLoss(clip_epsilon=0.2)(_forward([[0.5]]), batch)
        message = str(exc_info.value)
        assert "old_values" in message
        assert "nan" in message

    def test_zero_supervised_tokens_refused_never_zero(self) -> None:
        batch = _value_batch(masks=((0, 0), (0, 0)))
        with pytest.raises(SupervisionRefusal) as exc_info:
            ValueFunctionLoss()(_good_forward(), batch)
        message = str(exc_info.value)
        assert "0 supervised tokens" in message
        assert "2 offered rows" in message
