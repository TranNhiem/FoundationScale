"""Tests for the learned-value GAE and the TemporalAdvantageFn protocol seam."""

from __future__ import annotations

import math

import pytest

from foundationscale.rl.advantage import (
    AdvantageConfigRefusal,
    AdvantageRefusal,
    GroupNormalisedAdvantage,
    LearnedValueAdvantageEstimation,
    TemporalAdvantageFn,
)

# --- configuration: refused at construction, with the value named ---------


@pytest.mark.parametrize("field", ["gamma", "lambda_"])
@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), "0.5", None])
def test_discount_refused_outside_range_or_non_numeric(field: str, bad: object) -> None:
    # Both ends of [0, 1] are refused for BOTH fields, and NaN is refused by
    # the range check itself because every comparison against NaN is False.
    # The message must NAME the offending value, as the class docstring claims.
    with pytest.raises(AdvantageConfigRefusal) as exc_info:
        LearnedValueAdvantageEstimation(**{field: bad})
    message = str(exc_info.value)
    assert f"{field}={bad!r}" in message
    assert "[0.0, 1.0]" in message


def test_discount_boundary_values_zero_and_one_are_admitted() -> None:
    # [0, 1] is INCLUSIVE at both ends: gamma == 0.0 is "no bootstrapped
    # return" and lambda_ == 1.0 is full Monte-Carlo mixing. A mutation
    # tightening either comparison operator dies here, and the whiten
    # default stays a declared False.
    fn = LearnedValueAdvantageEstimation(gamma=0.0, lambda_=1.0)
    assert fn.gamma == 0.0
    assert fn.lambda_ == 1.0
    assert LearnedValueAdvantageEstimation().whiten is False


@pytest.mark.parametrize("bad", ["", 7])
def test_method_name_refuses_a_non_name(bad: object) -> None:
    with pytest.raises(AdvantageConfigRefusal) as exc_info:
        LearnedValueAdvantageEstimation(method_name=bad)
    assert "method_name" in str(exc_info.value)


@pytest.mark.parametrize("bad", [0, 1, "yes", None])
def test_whiten_refused_unless_a_declared_bool(bad: object) -> None:
    # Truthy stand-ins are refused: whitening is a recorded per-run policy
    # choice, so a manifest that wrote 1 must not silently mean True.
    with pytest.raises(AdvantageConfigRefusal) as exc_info:
        LearnedValueAdvantageEstimation(whiten=bad)
    assert f"whiten={bad!r}" in str(exc_info.value)


# --- the recursion, computed by hand ---------------------------------------


def test_monte_carlo_limit_telescopes_to_reward_minus_value() -> None:
    # gamma == lambda_ == 1, terminated. The recursion
    #   delta_t = r_t + V(t+1) - V(t),   A_t = delta_t + A_{t+1}
    # telescopes over a row: every interior V cancels between delta_t and the
    # carried A_{t+1}, leaving A_t = R - V(t) at every supervised position
    # (V(T) == 0.0 at the tail). Derived, not assumed, over values
    # (0.5, 0.25, 1.0) with R == 2.0:
    #   A_2 = (2.0 + 0.0  - 1.0)         = 1.0   == 2.0 - 1.0
    #   A_1 = (0.0 + 1.0  - 0.25) + A_2  = 1.75  == 2.0 - 0.25
    #   A_0 = (0.0 + 0.25 - 0.5)  + A_1  = 1.5   == 2.0 - 0.5
    out = LearnedValueAdvantageEstimation(gamma=1.0, lambda_=1.0).compute(
        prompt_ids=("p",),
        rewards=(2.0,),
        mask=((1, 1, 1),),
        values=((0.5, 0.25, 1.0),),
        terminated=(True,),
    )
    assert out.weights == ((1.5, 1.75, 1.0),)
    # The telescoping identity itself, pinned per position: a mutation that
    # fell back to a zero value baseline (all 2.0) or dropped the value
    # wiring entirely dies against these three equalities.
    for advantage, value in zip(out.weights[0], (0.5, 0.25, 1.0), strict=True):
        assert advantage == 2.0 - value
    assert out.used == out.offered == 1
    assert out.rows == (0,)
    assert out.method == "LearnedValueAdvantageEstimation"


def test_lambda_zero_is_one_step_td_and_each_advantage_is_its_delta() -> None:
    # gamma == 1.0, lambda_ == 0.0, so gamma * lambda_ == 0.0 kills A_{t+1}
    # and A_t == delta_t EXACTLY. Same row as the Monte-Carlo case; only the
    # tail delta carries reward:
    #   delta_2 = 2.0 + 0.0  - 1.0  =  1.0
    #   delta_1 = 0.0 + 1.0  - 0.25 =  0.75
    #   delta_0 = 0.0 + 0.25 - 0.5  = -0.25
    out = LearnedValueAdvantageEstimation(gamma=1.0, lambda_=0.0).compute(
        prompt_ids=("p",),
        rewards=(2.0,),
        mask=((1, 1, 1),),
        values=((0.5, 0.25, 1.0),),
        terminated=(True,),
    )
    assert out.weights == ((-0.25, 0.75, 1.0),)
    assert out.used == out.offered == 1


def test_intermediate_lambda_shows_the_geometric_mixing() -> None:
    # gamma == 0.5, lambda_ == 0.5, so gamma * lambda_ == 0.25. Same row
    # again; now each advantage is its delta plus a discounted share of the
    # carried tail, and the mixing is visible in the arithmetic:
    #   delta_2 = 2.0 + 0.5*0.0  - 1.0  =  1.0;   A_2 = 1.0
    #   delta_1 = 0.0 + 0.5*1.0  - 0.25 =  0.25;  A_1 = 0.25 + 0.25*1.0 = 0.5
    #   delta_0 = 0.0 + 0.5*0.25 - 0.5  = -0.375; A_0 = -0.375 + 0.25*0.5
    #                                               = -0.25
    # A_0 differs from both delta_0 (-0.375) and the undiscounted sum
    # delta_0 + delta_1 + delta_2 (0.25): the residual 0.125 is exactly
    # 0.25 * 0.5, the geometric weight on the carried tail. A mutation that
    # used gamma alone, lambda_ alone, or their sum dies on this figure.
    out = LearnedValueAdvantageEstimation(gamma=0.5, lambda_=0.5).compute(
        prompt_ids=("p",),
        rewards=(2.0,),
        mask=((1, 1, 1),),
        values=((0.5, 0.25, 1.0),),
        terminated=(True,),
    )
    assert out.weights == ((-0.25, 0.5, 1.0),)
    assert out.used == out.offered == 1


# --- the episode boundary: declared, not guessed ---------------------------


def test_terminal_and_truncated_tails_give_different_numbers() -> None:
    # THE discriminating test for bootstrapping. gamma == 1.0, lambda_ == 0.0,
    # identical row data, and ONLY the `terminated` declaration flips.
    # Terminal reads V(T) == 0.0: delta_2 = 2.0 + 0.0 - 1.0 = 1.0.
    # Truncated bootstraps from the last value estimate, V(T) == 1.0:
    #               delta_2 = 2.0 + 1.0 - 1.0 = 2.0.
    # Interior deltas coincide either way, so ONLY the tail moves -- a
    # binding that hard-codes one tail, or ignores the declaration, is
    # killed by whichever arm it does not serve.
    terminal = LearnedValueAdvantageEstimation(gamma=1.0, lambda_=0.0).compute(
        prompt_ids=("p",),
        rewards=(2.0,),
        mask=((1, 1, 1),),
        values=((0.5, 0.25, 1.0),),
        terminated=(True,),
    )
    truncated = LearnedValueAdvantageEstimation(gamma=1.0, lambda_=0.0).compute(
        prompt_ids=("p",),
        rewards=(2.0,),
        mask=((1, 1, 1),),
        values=((0.5, 0.25, 1.0),),
        terminated=(False,),
    )
    assert terminal.weights == ((-0.25, 0.75, 1.0),)
    assert truncated.weights == ((-0.25, 0.75, 2.0),)
    assert terminal.weights != truncated.weights


@pytest.mark.parametrize("bad", [0, 1, "True", None])
def test_non_bool_terminated_entry_refused_with_row_named(bad: object) -> None:
    # The tail must be DECLARED: 1 is not True here, because a truthy
    # stand-in would guess the episode boundary the contract refuses to
    # guess, biasing every advantage in the row.
    with pytest.raises(AdvantageRefusal) as exc_info:
        LearnedValueAdvantageEstimation().compute(
            prompt_ids=("p",),
            rewards=(1.0,),
            mask=((1,),),
            values=((0.0,),),
            terminated=(bad,),
        )
    assert f"terminated[0] is {bad!r}" in str(exc_info.value)


# --- masking: literal zeros, and skipped by the recursion ------------------


def test_masked_positions_are_skipped_by_the_recursion_and_written_zero() -> None:
    # Row 0 masks its middle position, which carries a DELIBERATELY ABSURD
    # value estimate (999.0): any recursion that consulted it would smear it
    # into A_0. Row 1 is the same sequence with the masked position DELETED.
    # gamma == lambda_ == 1 telescopes, so A_t = R - V(t) over the supervised
    # positions in order:
    #   row 0: A_2 = 2.0 - 1.0 = 1.0;  A_0 = 2.0 - 0.5 = 1.5; middle: 0.0
    #   row 1: A_1 = 2.0 - 1.0 = 1.0;  A_0 = 1.5
    # Equality across the two rows IS the skip assertion: the masked position
    # neither carries an advantage nor feeds one to its neighbours.
    out = LearnedValueAdvantageEstimation(gamma=1.0, lambda_=1.0).compute(
        prompt_ids=("p", "q"),
        rewards=(2.0, 2.0),
        mask=((1, 0, 1), (1, 1)),
        values=((0.5, 999.0, 1.0), (0.5, 1.0)),
        terminated=(True, True),
    )
    assert out.weights == ((1.5, 0.0, 1.0), (1.5, 1.0))
    assert out.weights[0][1] == 0.0  # a literal mask zero, never a measured one
    assert out.weights[0][0] == out.weights[1][0]
    assert out.weights[0][2] == out.weights[1][1]
    assert out.used == out.offered == 2
    assert out.rows == (0, 1)


# --- structural refusals: ragged shapes and empty denominators -------------


def test_row_with_zero_supervised_tokens_is_refused() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        LearnedValueAdvantageEstimation().compute(
            prompt_ids=("a", "b"),
            rewards=(1.0, 2.0),
            mask=((0, 0), (1,)),
            values=((0.1, 0.2), (0.3,)),
            terminated=(True, False),
        )
    message = str(exc_info.value)
    assert "mask row 0" in message
    assert "0 of 2" in message


def test_empty_batch_is_refused_as_vacuous() -> None:
    # Statistics over zero supervised elements are UNMEASURED, never zero, so
    # the empty batch refuses rather than returning an empty result.
    with pytest.raises(AdvantageRefusal) as exc_info:
        LearnedValueAdvantageEstimation().compute(
            prompt_ids=(), rewards=(), mask=(), values=(), terminated=()
        )
    assert "0 samples" in str(exc_info.value)


def test_values_and_mask_sample_count_disagreement_refused() -> None:
    # One value row for two mask rows: a silent mismatch would subtract one
    # row's baseline from another row's rewards.
    with pytest.raises(AdvantageRefusal) as exc_info:
        LearnedValueAdvantageEstimation().compute(
            prompt_ids=("a", "b"),
            rewards=(1.0, 2.0),
            mask=((1,), (1,)),
            values=((0.1,),),
            terminated=(True, True),
        )
    message = str(exc_info.value)
    assert "1 value rows" in message
    assert "2 mask rows" in message


def test_values_row_shorter_than_its_mask_row_refused() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        LearnedValueAdvantageEstimation().compute(
            prompt_ids=("a",),
            rewards=(1.0,),
            mask=((1, 1, 1),),
            values=((0.1, 0.2),),
            terminated=(True,),
        )
    message = str(exc_info.value)
    assert "values row 0 has 2 entries" in message
    assert "mask row 0 has 3 positions" in message


def test_terminated_and_rewards_count_disagreement_refused() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        LearnedValueAdvantageEstimation().compute(
            prompt_ids=("a", "b"),
            rewards=(1.0, 2.0),
            mask=((1,), (1,)),
            values=((0.1,), (0.2,)),
            terminated=(True,),
        )
    message = str(exc_info.value)
    assert "1 tail declarations" in message
    assert "2 mask rows" in message


def test_non_finite_value_estimate_refused_with_coordinates_named() -> None:
    # One non-finite V(t) enters every delta from the row's tail back to t,
    # so it is refused outright, with row and position named.
    with pytest.raises(AdvantageRefusal) as exc_info:
        LearnedValueAdvantageEstimation().compute(
            prompt_ids=("a",),
            rewards=(1.0,),
            mask=((1, 1),),
            values=((0.5, float("inf")),),
            terminated=(True,),
        )
    message = str(exc_info.value)
    assert "row 0, position 1" in message
    assert "not finite" in message


def test_unconvertible_value_estimate_refused_with_coordinates_named() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        LearnedValueAdvantageEstimation().compute(
            prompt_ids=("a",),
            rewards=(1.0,),
            mask=((1, 1),),
            values=((0.5, "wide"),),
            terminated=(True,),
        )
    assert "row 0, position 1" in str(exc_info.value)


# --- whitening: supervised-only statistics, zero-spread abstention ---------


def test_whiten_normalises_over_supervised_positions_only() -> None:
    # gamma == 1.0, lambda_ == 0.0, terminated: each advantage is its delta.
    #   row 0: mask (1, 1, 0), values (0.0, 0.0, 9.0), reward 4.0
    #     A_1 = 4.0 - 0.0 = 4.0;  A_0 = 0.0 - 0.0 = 0.0   [masked: literal 0.0]
    #   row 1: mask (1, 1), values (1.0, 1.0), reward 0.0
    #     A_1 = 0.0 - 1.0 = -1.0; A_0 = 1.0 - 1.0 = 0.0
    # The supervised advantages are (0.0, 4.0, 0.0, -1.0): mean 0.75 and
    # population variance (0.5625 + 10.5625 + 0.5625 + 3.0625) / 4 == 3.6875.
    # Had the masked literal 0.0 entered the statistic the mean would be 0.6;
    # the masked position is pinned OUT of it by the exact figures below.
    spread = math.sqrt(3.6875)
    out = LearnedValueAdvantageEstimation(gamma=1.0, lambda_=0.0, whiten=True).compute(
        prompt_ids=("a", "b"),
        rewards=(4.0, 0.0),
        mask=((1, 1, 0), (1, 1)),
        values=((0.0, 0.0, 9.0), (1.0, 1.0)),
        terminated=(True, True),
    )
    assert out.weights[0] == pytest.approx((-0.75 / spread, 3.25 / spread, 0.0))
    assert out.weights[0][2] == 0.0  # still the literal mask zero afterwards
    assert out.weights[1] == pytest.approx((-0.75 / spread, -1.75 / spread))
    # Output-side check over the supervised positions only: mean ~0 and
    # population std ~1, which only holds when the statistic excluded the
    # masked zero at position (0, 2).
    supervised = out.weights[0][:2] + out.weights[1]
    assert sum(supervised) / len(supervised) == pytest.approx(0.0)
    assert sum(a * a for a in supervised) / len(supervised) == pytest.approx(1.0)
    assert out.used == out.offered == 2
    assert out.rows == (0, 1)


def test_whiten_abstains_when_the_supervised_spread_is_zero() -> None:
    # gamma == 1.0, lambda_ == 0.0, terminated, and inputs crafted so every
    # delta lands on exactly 1.0:
    #   row 0: mask (1, 0, 1), values (1.0, 99.0, 2.0), reward 3.0
    #     delta_2 = 3.0 - 2.0 = 1.0;  delta_0 = 2.0 - 1.0 = 1.0
    #   row 1: mask (1, 1), values (4.0, 5.0), reward 6.0
    #     delta_1 = 6.0 - 5.0 = 1.0;  delta_0 = 5.0 - 4.0 = 1.0
    # The population std over the supervised advantages is EXACTLY 0.0, so
    # whitening ABSTAINS: nothing is divided by nothing and no all-zero row
    # is written as if it were a measurement. The RAW estimates must return
    # unchanged, asserted exactly -- a mutation dividing by the zero std, or
    # writing zeros, dies on these two tuples.
    out = LearnedValueAdvantageEstimation(gamma=1.0, lambda_=0.0, whiten=True).compute(
        prompt_ids=("a", "b"),
        rewards=(3.0, 6.0),
        mask=((1, 0, 1), (1, 1)),
        values=((1.0, 99.0, 2.0), (4.0, 5.0)),
        terminated=(True, True),
    )
    assert out.weights == ((1.0, 0.0, 1.0), (1.0, 1.0))
    assert out.used == out.offered == 2


def test_rows_are_the_identity_because_nothing_is_excluded() -> None:
    # As with the zero-baseline sibling: used == offered by construction and
    # the index list is the full range, stated so a caller never has to know
    # which advantage functions compact and which do not.
    out = LearnedValueAdvantageEstimation().compute(
        prompt_ids=("a", "b", "c"),
        rewards=(1.0, 2.0, 3.0),
        mask=((1, 1), (1,), (1, 1, 1)),
        values=((0.0, 0.0), (0.0,), (0.0, 0.0, 0.0)),
        terminated=(True, True, True),
    )
    assert out.used == out.offered == 3
    assert out.rows == (0, 1, 2)
    assert out.rewards.count == 3
    assert out.rewards.maximum == 3.0


def test_method_name_flows_into_the_result() -> None:
    out = LearnedValueAdvantageEstimation(method_name="lvgae_v2").compute(
        prompt_ids=("p",),
        rewards=(2.0,),
        mask=((1, 1),),
        values=((0.5, 0.5),),
        terminated=(True,),
    )
    assert out.method == "lvgae_v2"
    assert out.used == out.offered == 1


# --- the TemporalAdvantageFn seam ------------------------------------------


def test_temporal_advantage_fn_is_a_runtime_checkable_protocol() -> None:
    assert isinstance(LearnedValueAdvantageEstimation(), TemporalAdvantageFn)
    assert not isinstance(object(), TemporalAdvantageFn)


def test_existing_advantage_fns_pass_isinstance_then_fail_at_the_call() -> None:
    """WHAT IS CLAIMED: the documented hole, pinned. ``@runtime_checkable``
    checks METHOD PRESENCE, never signatures, so an existing
    :class:`AdvantageFn` implementation satisfies ``isinstance`` against
    :class:`TemporalAdvantageFn` even though it accepts neither ``values``
    nor ``terminated`` -- and actually calling it with the temporal keywords
    raises ``TypeError`` at the call site. This is exactly the failure the
    ``TemporalAdvantageFn`` docstring asserts, and any refactor that unified
    the signatures would have to delete this test on purpose rather than
    discover the hole had closed quietly. WHAT IS NOT CLAIMED: that the
    ``isinstance`` success carries any information about conformance -- it
    does not; that is the entire reason ``TemporalAdvantageFn`` does not
    inherit from ``AdvantageFn``.
    """
    grpo = GroupNormalisedAdvantage()
    assert isinstance(grpo, TemporalAdvantageFn)
    with pytest.raises(TypeError) as exc_info:
        grpo.compute(
            prompt_ids=("a", "b"),
            rewards=(1.0, 2.0),
            mask=((1,), (1,)),
            values=((0.0,), (0.0,)),
            terminated=(True, True),
        )
    assert "values" in str(exc_info.value)
