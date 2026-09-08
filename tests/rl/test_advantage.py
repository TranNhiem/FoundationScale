"""Tests for the stage-3 advantage functions."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest

# #327: this module is the ONE place that imports both planes' RewardStats, and it
# does so in order to pin that they stay apart. It is not a counterexample to the
# separation -- no SOURCE file imports both.
from foundationscale.gates.objective_gates import RewardStats as GateRewardStats
from foundationscale.rl.advantage import (
    AdvantageConfigRefusal,
    AdvantageFn,
    AdvantageRefusal,
    AdvantageResult,
    GeneralisedAdvantageEstimation,
    GroupNormalisedAdvantage,
    LeaveOneOutAdvantage,
    RewardStats,
)


def _result_kwargs() -> dict[str, Any]:
    return {
        "weights": ((1.0, 0.0),),
        "rewards": RewardStats.over((2.0,)),
        "used": 1,
        "offered": 4,
        "method": "m",
        "rows": (0,),
    }


def test_reward_stats_population_std_not_sample_std() -> None:
    # (1, 2, 3, 4): mean 2.5; squared deviations 2.25, 0.25, 0.25, 2.25.
    # POPULATION variance divides by n == 4, giving 1.25; the SAMPLE divisor
    # n - 1 would give 5/3. The contract names the choice in the docstring
    # because the two differ and a reader will assume one.
    stats = RewardStats.over((1.0, 2.0, 3.0, 4.0))
    assert stats.count == 4
    assert stats.mean == 2.5
    assert stats.std == pytest.approx(math.sqrt(1.25))
    assert stats.std != pytest.approx(math.sqrt(5.0 / 3.0))
    assert stats.minimum == 1.0
    assert stats.maximum == 4.0


def test_reward_stats_over_a_single_value_has_zero_spread() -> None:
    stats = RewardStats.over((3.5,))
    assert stats.std == 0.0
    assert stats.minimum == stats.maximum == 3.5


def test_reward_stats_over_an_empty_sequence_refused() -> None:
    # Nothing used means nothing measured: an UNMEASURED state, never a
    # zeroed pass.
    with pytest.raises(AdvantageRefusal) as exc_info:
        RewardStats.over(())
    assert "empty" in str(exc_info.value)


def test_reward_stats_unconvertible_value_refused() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        RewardStats.over((1.0, object()))
    assert "position 1" in str(exc_info.value)


def test_reward_stats_non_finite_value_refused() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        RewardStats.over((float("nan"),))
    message = str(exc_info.value)
    assert "position 0" in message
    assert "not finite" in message


def test_reward_stats_zero_count_refused_at_construction() -> None:
    # A hand-built zero-count value object would claim a denominator of
    # nothing, so construction refuses as firmly as over(()) does.
    with pytest.raises(AdvantageRefusal) as exc_info:
        RewardStats(count=0, mean=0.0, std=0.0, minimum=0.0, maximum=0.0)
    assert "count=0" in str(exc_info.value)


def test_advantage_result_freezes_rows_and_claims_what_it_used() -> None:
    result = AdvantageResult(
        weights=[[1.0, 0.0]],  # type: ignore[arg-type]  # the freeze is under test
        rewards=RewardStats.over((2.0,)),
        used=1,
        offered=4,
        method="m",
        rows=[0],  # type: ignore[arg-type]  # the freeze is under test
    )
    assert result.weights == ((1.0, 0.0),)
    assert result.rows == (0,)
    assert result.used == 1
    assert result.offered == 4
    assert result.rewards.count == 1


@pytest.mark.parametrize("bad_method", ["", 7])
def test_advantage_result_refuses_a_non_name_method(bad_method: object) -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        AdvantageResult(**{**_result_kwargs(), "method": bad_method})  # type: ignore[arg-type]
    assert "method" in str(exc_info.value)


def test_advantage_result_refused_when_used_exceeds_offered() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        AdvantageResult(**{**_result_kwargs(), "used": 5})
    message = str(exc_info.value)
    assert "used=5" in message
    assert "offered=4" in message


def test_advantage_result_refused_when_weight_rows_do_not_match_used() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        AdvantageResult(**{**_result_kwargs(), "used": 2})
    message = str(exc_info.value)
    assert "len(weights)=1" in message
    assert "used=2" in message


def test_advantage_result_refused_when_stats_describe_other_samples() -> None:
    # Two used rows but statistics over ONE sample: the result would claim a
    # denominator it did not use.
    with pytest.raises(AdvantageRefusal) as exc_info:
        AdvantageResult(
            **{
                **_result_kwargs(),
                "weights": ((1.0,), (1.0,)),
                "used": 2,
                "offered": 2,
            }
        )
    message = str(exc_info.value)
    assert "rewards.count=1" in message
    assert "used=2" in message


def test_advantage_result_refused_when_rows_does_not_name_every_used_sample() -> None:
    # Two weight rows but one named index: the second row would be
    # unattributable, which is the whole failure `rows` exists to prevent.
    with pytest.raises(AdvantageRefusal) as exc_info:
        AdvantageResult(
            **{
                **_result_kwargs(),
                "weights": ((1.0,), (1.0,)),
                "rewards": RewardStats.over((2.0, 4.0)),
                "used": 2,
                "rows": (0,),
            }
        )
    message = str(exc_info.value)
    assert "len(rows)=1" in message
    assert "used=2" in message


@pytest.mark.parametrize("bad_index", [True, 1.0, "0", None])
def test_advantage_result_refused_when_a_row_index_is_not_an_int(
    bad_index: object,
) -> None:
    # `True` is in the set deliberately: bool is a subclass of int, so an
    # isinstance check alone would let a flag through as row 1.
    with pytest.raises(AdvantageRefusal) as exc_info:
        AdvantageResult(**{**_result_kwargs(), "rows": (bad_index,)})
    assert "rows[0]" in str(exc_info.value)


@pytest.mark.parametrize("bad_index", [-1, 4, 99])
def test_advantage_result_refused_when_a_row_index_was_never_offered(
    bad_index: int,
) -> None:
    # offered == 4, so the valid range is [0, 4).
    with pytest.raises(AdvantageRefusal) as exc_info:
        AdvantageResult(**{**_result_kwargs(), "rows": (bad_index,)})
    message = str(exc_info.value)
    assert f"rows[0]={bad_index}" in message
    assert "[0, 4)" in message


@pytest.mark.parametrize("bad_rows", [(1, 1), (2, 0)])
def test_advantage_result_refused_when_rows_is_not_strictly_increasing(
    bad_rows: tuple[int, ...],
) -> None:
    # Duplicate and descending both fail: `rows` must be a SUBSEQUENCE of the
    # offered batch, so a caller may zip weights against it without also
    # having to trust that the producer preserved batch order.
    with pytest.raises(AdvantageRefusal) as exc_info:
        AdvantageResult(
            **{
                **_result_kwargs(),
                "weights": ((1.0,), (1.0,)),
                "rewards": RewardStats.over((2.0, 4.0)),
                "used": 2,
                "rows": bad_rows,
            }
        )
    assert "strictly increasing" in str(exc_info.value)


def test_advantage_fn_is_a_runtime_checkable_protocol() -> None:
    assert isinstance(GroupNormalisedAdvantage(), AdvantageFn)
    assert isinstance(LeaveOneOutAdvantage(), AdvantageFn)
    assert isinstance(GeneralisedAdvantageEstimation(), AdvantageFn)
    assert not isinstance(object(), AdvantageFn)


def test_group_normalised_hand_computed_and_degenerate_group_gap() -> None:
    # Group "a" rewards (1, 3, 5): mean 3.0; population variance
    # ((1-3)^2 + (3-3)^2 + (5-3)^2) / 3 = 8/3, so std == sqrt(8/3) and
    #   adv(1) = (1-3)/sqrt(8/3) = -sqrt(3/2)
    #   adv(3) = 0.0
    #   adv(5) = (5-3)/sqrt(8/3) = +sqrt(3/2)
    # Group "b" rewards (2, 2): std == 0.0, NO signal. The pair leaves the
    # result entirely rather than sitting in it as zero rows, so the gap
    # offered - used == 2 is the visible record of the degenerate group.
    spread = math.sqrt(3.0 / 2.0)
    out = GroupNormalisedAdvantage().compute(
        prompt_ids=("a", "a", "a", "b", "b"),
        rewards=(1.0, 3.0, 5.0, 2.0, 2.0),
        mask=((1, 1, 0), (1, 0), (1,), (1, 1), (1, 1)),
    )
    assert out.weights[0] == pytest.approx((-spread, -spread, 0.0))
    assert out.weights[1] == (0.0, 0.0)  # the measured-zero advantage row
    assert out.weights[2] == pytest.approx((spread,))
    assert len(out.weights) == 3
    assert out.used == 3
    assert out.offered == 5
    assert out.offered - out.used == 2  # the degenerate group, kept visible
    assert out.method == "GroupNormalisedAdvantage"
    # The statistics summarise the USED samples only -- the (2, 2) pair is
    # nowhere in the denominator.
    assert out.rewards.count == 3
    assert out.rewards.mean == 3.0
    assert out.rewards.std == pytest.approx(math.sqrt(8.0 / 3.0))
    assert out.rewards.minimum == 1.0
    assert out.rewards.maximum == 5.0


def test_group_normalised_singleton_groups_leave_the_result() -> None:
    # "a" and "c" are singletons: a group of 1 cannot be normalised, so both
    # leave. "b" rewards (2, 4): mean 3.0, std 1.0, advantages -1.0, +1.0.
    out = GroupNormalisedAdvantage().compute(
        prompt_ids=("a", "b", "b", "c"),
        rewards=(1.0, 2.0, 4.0, 9.0),
        mask=((1,), (1,), (1,), (1,)),
    )
    assert out.weights == ((-1.0,), (1.0,))
    assert out.used == 2
    assert out.offered == 4
    assert out.offered - out.used == 2


def test_group_normalised_rows_name_the_survivors_of_an_interleaved_drop() -> None:
    # The case a COUNT cannot express. Groups are interleaved, so the dropped
    # rows are not a suffix: "a" sits at offered-rows 0 and 2, "b" (constant,
    # no signal) at 1 and 3. `weights` is compacted to two rows, so a caller
    # that zipped it positionally against the batch would apply group "a"'s
    # advantages to rows 0 and 1 -- wrong sequence, no exception. `rows` is
    # what makes that impossible: it says 0 and 2.
    out = GroupNormalisedAdvantage().compute(
        prompt_ids=("a", "b", "a", "b"),
        rewards=(1.0, 5.0, 3.0, 5.0),
        mask=((1,), (1,), (1,), (1,)),
    )
    assert out.weights == ((-1.0,), (1.0,))
    assert out.used == 2
    assert out.offered == 4
    assert out.rows == (0, 2)  # NOT (0, 1)


def test_leave_one_out_rows_name_the_survivors_of_an_interleaved_drop() -> None:
    # Same shape for RLOO, whose exclusion rule differs (singleton groups
    # have no leave-one-out baseline) but whose compaction hazard is identical.
    out = LeaveOneOutAdvantage().compute(
        prompt_ids=("solo", "pair", "other", "pair"),
        rewards=(9.0, 2.0, 7.0, 4.0),
        mask=((1,), (1,), (1,), (1,)),
    )
    assert out.used == 2
    assert out.offered == 4
    assert out.rows == (1, 3)  # the two "pair" members, not (0, 1)


def test_gae_rows_are_the_identity_because_gae_excludes_nothing() -> None:
    # GAE has no grouping and so no exclusion rule: used == offered and the
    # index list is the full range. Stated rather than defaulted, so a caller
    # never has to know which advantage functions compact and which do not.
    out = GeneralisedAdvantageEstimation().compute(
        prompt_ids=("a", "b", "c"),
        rewards=(1.0, 2.0, 3.0),
        mask=((1, 1), (1,), (1, 1, 1)),
    )
    assert out.used == out.offered == 3
    assert out.rows == (0, 1, 2)


def test_group_normalised_min_group_size_is_configurable() -> None:
    # min_group_size=3: group "b" (2 samples) now leaves too. Group "a"
    # rewards (0, 1, 2): mean 1.0, population std sqrt(2/3), so the outer
    # advantages are +-1/sqrt(2/3) == +-sqrt(3/2).
    spread = math.sqrt(3.0 / 2.0)
    out = GroupNormalisedAdvantage(min_group_size=3).compute(
        prompt_ids=("a", "a", "a", "b", "b"),
        rewards=(0.0, 1.0, 2.0, 5.0, 7.0),
        mask=((1,), (1,), (1,), (1,), (1,)),
    )
    assert out.weights[0] == pytest.approx((-spread,))
    assert out.weights[1] == (0.0,)
    assert out.weights[2] == pytest.approx((spread,))
    assert out.used == 3
    assert out.offered == 5


def test_group_normalised_refuses_when_every_group_has_no_signal() -> None:
    # A constant group plus a singleton: nothing usable survived, so
    # RewardStats.over(()) refuses -- an empty denominator is UNMEASURED,
    # never a zeroed pass.
    with pytest.raises(AdvantageRefusal) as exc_info:
        GroupNormalisedAdvantage().compute(
            prompt_ids=("a", "a", "b"),
            rewards=(1.0, 1.0, 2.0),
            mask=((1,), (1,), (1,)),
        )
    assert "nothing used" in str(exc_info.value)


def test_method_name_flows_into_the_result() -> None:
    out = GroupNormalisedAdvantage(method_name="grpo_v2").compute(
        prompt_ids=("p", "p"),
        rewards=(0.0, 2.0),
        mask=((1,), (1,)),
    )
    assert out.method == "grpo_v2"
    assert out.weights == ((-1.0,), (1.0,))


def test_leave_one_out_hand_computed_baseline() -> None:
    # Group rewards (1, 3, 5); each baseline is the mean of the OTHERS:
    #   r=1: baseline (3+5)/2 = 4.0, advantage 1.0 - 4.0 = -3.0
    #   r=3: baseline (1+5)/2 = 3.0, advantage 3.0 - 3.0 =  0.0
    #   r=5: baseline (1+3)/2 = 2.0, advantage 5.0 - 2.0 = +3.0
    out = LeaveOneOutAdvantage().compute(
        prompt_ids=("p", "p", "p"),
        rewards=(1.0, 3.0, 5.0),
        mask=((1, 1), (1, 0), (0, 1)),
    )
    assert out.weights == ((-3.0, -3.0), (0.0, 0.0), (0.0, 3.0))
    assert out.used == out.offered == 3
    assert out.method == "LeaveOneOutAdvantage"
    assert out.rewards.count == 3
    assert out.rewards.mean == 3.0


def test_leave_one_out_zero_variance_group_is_a_measured_zero_not_an_exclusion() -> None:
    # A constant group leaves each sample EXACTLY equal to its leave-one-out
    # baseline, so the 0.0 advantage is a genuinely measured reading ("this
    # sample is exactly average among its peers") -- unlike GRPO, where the
    # same group has no signal at all. Measured zeros stay in the
    # denominator: used == offered.
    out = LeaveOneOutAdvantage().compute(
        prompt_ids=("p", "p"),
        rewards=(2.0, 2.0),
        mask=((1,), (1,)),
    )
    assert out.weights == ((0.0,), (0.0,))
    assert out.used == out.offered == 2


def test_leave_one_out_singleton_groups_leave_the_result() -> None:
    # "a" and "c" are singletons: a leave-one-out baseline needs at least
    # one OTHER sample, so both leave. "b" rewards (2, 4): baselines 4.0
    # and 2.0, advantages -2.0, +2.0.
    out = LeaveOneOutAdvantage().compute(
        prompt_ids=("a", "b", "b", "c"),
        rewards=(1.0, 2.0, 4.0, 9.0),
        mask=((1,), (1,), (1,), (1,)),
    )
    assert out.weights == ((-2.0,), (2.0,))
    assert out.used == 2
    assert out.offered == 4
    assert out.offered - out.used == 2


def test_gae_unit_discounts_reduce_to_reward_to_go() -> None:
    # With gamma == lam == 1 the recursion is A_t = r_t + A_{t+1}: the
    # reward-to-go. That is the right reduction to check because
    # reward-to-go is the undiscounted limit of GAE with a zero value
    # baseline, so any slip in the recursion direction or the terminal
    # placement shows up against closed-form arithmetic. ALL the reward
    # mass sits on the last unmasked position, so every unmasked position's
    # reward-to-go is the full terminal reward. The mask arrives as bools,
    # exercising the bool-before-int admission path.
    out = GeneralisedAdvantageEstimation(gamma=1.0, lam=1.0).compute(
        prompt_ids=("p",),
        rewards=(2.5,),
        mask=((True, False, True, True),),
    )
    assert out.weights == ((2.5, 0.0, 2.5, 2.5),)
    assert out.used == out.offered == 1
    assert out.method == "GeneralisedAdvantageEstimation"


def test_gae_backward_recursion_hand_computed() -> None:
    # gamma == lam == 0.5, so gamma * lam == 0.25 and (with V == 0)
    # A_t = r_t + 0.25 * A_{t+1} over the unmasked positions, right to left.
    # Row 0: mask (1, 1, 1), terminal reward 4.0 on position 2:
    #   A_2 = 4.0
    #   A_1 = 0.0 + 0.25 * 4.0 = 1.0
    #   A_0 = 0.0 + 0.25 * 1.0 = 0.25
    # Row 1: mask (1, 0, 1), terminal reward 2.0 on position 2 (the LAST
    # unmasked one). The chain runs over unmasked positions only, so the
    # masked middle position is skipped by the recursion AND written 0.0:
    #   A_2 = 2.0
    #   A_0 = 0.0 + 0.25 * 2.0 = 0.5
    out = GeneralisedAdvantageEstimation(gamma=0.5, lam=0.5).compute(
        prompt_ids=("p", "q"),
        rewards=(4.0, 2.0),
        mask=((1, 1, 1), (1, 0, 1)),
    )
    assert out.weights == ((0.25, 1.0, 4.0), (0.5, 0.0, 2.0))
    assert out.used == out.offered == 2
    # 4.0 and 2.0: mean 3.0, population variance (1 + 1) / 2 == 1.0.
    assert out.rewards == RewardStats(count=2, mean=3.0, std=1.0, minimum=2.0, maximum=4.0)


@pytest.mark.parametrize(
    "cls",
    [GroupNormalisedAdvantage, LeaveOneOutAdvantage, GeneralisedAdvantageEstimation],
)
@pytest.mark.parametrize("bad", ["", 7])
def test_every_method_name_field_refuses_a_non_name(cls: Any, bad: object) -> None:
    # Parametrised over all THREE implementations: a method_name field added
    # without the guard is the case this catches, and one representative
    # class would not catch it.
    with pytest.raises(AdvantageConfigRefusal) as exc_info:
        cls(method_name=bad)
    assert "method_name" in str(exc_info.value)


@pytest.mark.parametrize("cls", [GroupNormalisedAdvantage, LeaveOneOutAdvantage])
@pytest.mark.parametrize("bad", [0, 1, 2.5, "2", None])
def test_min_group_size_below_two_refused(cls: Any, bad: object) -> None:
    with pytest.raises(AdvantageConfigRefusal) as exc_info:
        cls(min_group_size=bad)
    assert "min_group_size" in str(exc_info.value)


@pytest.mark.parametrize("field", ["gamma", "lam"])
@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), "0.5"])
def test_gae_discount_out_of_range_refused(field: str, bad: object) -> None:
    # NaN is refused by the range check itself: every comparison against NaN
    # is False, so "not (0.0 <= nan <= 1.0)" holds.
    with pytest.raises(AdvantageConfigRefusal) as exc_info:
        GeneralisedAdvantageEstimation(**{field: bad})  # type: ignore[arg-type]
    assert field in str(exc_info.value)


@pytest.mark.parametrize(
    "cls",
    [GroupNormalisedAdvantage, LeaveOneOutAdvantage, GeneralisedAdvantageEstimation],
)
def test_length_disagreement_names_all_three_lengths(cls: Any) -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        cls().compute(prompt_ids=("a", "b"), rewards=(1.0, 2.0), mask=((1,),))
    message = str(exc_info.value)
    assert "2 prompt ids" in message
    assert "2 rewards" in message
    assert "1 mask rows" in message


def test_zero_samples_refused() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        GroupNormalisedAdvantage().compute(prompt_ids=(), rewards=(), mask=())
    assert "0 samples" in str(exc_info.value)


def test_unconvertible_reward_refused() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        GroupNormalisedAdvantage().compute(
            prompt_ids=("a", "b"), rewards=(1.0, object()), mask=((1,), (1,))
        )
    assert "row 1, position 1" in str(exc_info.value)


def test_non_finite_reward_refused() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        GroupNormalisedAdvantage().compute(prompt_ids=("a",), rewards=(float("nan"),), mask=((1,),))
    message = str(exc_info.value)
    assert "row 0, position 0" in message
    assert "nan" in message
    assert "not finite" in message


def test_fractional_mask_entry_refused() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        GroupNormalisedAdvantage().compute(prompt_ids=("a",), rewards=(1.0,), mask=((1, 0.5),))
    message = str(exc_info.value)
    assert "row 0, position 1" in message
    assert "0 or 1" in message


def test_unconvertible_mask_entry_refused() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        GroupNormalisedAdvantage().compute(prompt_ids=("a",), rewards=(1.0,), mask=((1, object()),))
    assert "row 0, position 1" in str(exc_info.value)


def test_empty_mask_row_refused() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        GroupNormalisedAdvantage().compute(
            prompt_ids=("a", "b"), rewards=(1.0, 2.0), mask=((), (1,))
        )
    assert "mask row 0 is empty" in str(exc_info.value)


def test_all_zero_mask_row_refused() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        GroupNormalisedAdvantage().compute(prompt_ids=("a",), rewards=(1.0,), mask=((0, 0),))
    message = str(exc_info.value)
    assert "row 0" in message
    assert "0 of 2" in message


def test_non_iterable_rewards_refused() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        GroupNormalisedAdvantage().compute(
            prompt_ids=("a",),
            rewards=7,
            mask=((1,),),  # type: ignore[arg-type]
        )
    assert "rewards" in str(exc_info.value)


def test_non_iterable_mask_row_refused() -> None:
    with pytest.raises(AdvantageRefusal) as exc_info:
        GroupNormalisedAdvantage().compute(
            prompt_ids=("a", "b"),
            rewards=(1.0, 2.0),
            mask=((1, 1), 7),  # type: ignore[arg-type]
        )
    assert "mask row 1" in str(exc_info.value)


def test_unhashable_prompt_id_refused() -> None:
    # Prompt ids are grouping keys; a list cannot hash, and silently
    # skipping the row would hide a sample from the denominator.
    with pytest.raises(AdvantageRefusal) as exc_info:
        GroupNormalisedAdvantage().compute(
            prompt_ids=(["a", "b"], ["a"]),  # type: ignore[arg-type]
            rewards=(1.0, 2.0),
            mask=((1,), (1,)),
        )
    assert "row 0" in str(exc_info.value)


# --- #327: the two RewardStats classes, and why they must stay apart ----------


def test_the_two_rewardstats_classes_are_distinct_types() -> None:
    """One name, two classes -- and NOT one type reached by two import paths.

    A tidy-up that unified them would leave this test failing rather than
    silently merging two contracts.
    """
    assert GateRewardStats is not RewardStats
    assert not issubclass(GateRewardStats, RewardStats)
    assert not issubclass(RewardStats, GateRewardStats)


def test_neither_rewardstats_field_set_is_a_subset_of_the_other() -> None:
    """WHAT IS CLAIMED: substitution between the planes fails loudly at
    construction because the vocabularies are disjoint on the fields that
    matter -- the count-and-extremes keywords of one class are rejected by the
    other, so dropping one in for the other raises TypeError at the call site
    instead of smuggling a wrong-plane summary through. WHAT IS NOT CLAIMED:
    anything about whether the two summarise the same samples -- they do not;
    one summarises the samples inspected AT THE GATE POINT, the other the
    samples an advantage function ACTUALLY USED.
    """
    gate_fields = {f.name for f in dataclasses.fields(GateRewardStats)}
    rl_fields = {f.name for f in dataclasses.fields(RewardStats)}

    assert gate_fields != rl_fields
    # Both directions separately, so a failure names which direction broke
    # instead of collapsing into one undifferentiated subset assertion.
    assert not gate_fields <= rl_fields
    assert not rl_fields <= gate_fields

    # Named, not counted: a test that compared only sizes would pass through a
    # rename that swapped one name for another, which is the exact edit that
    # would make substitution start succeeding silently.
    assert {"n", "min", "max"} <= gate_fields
    assert {"n", "min", "max"}.isdisjoint(rl_fields)
    assert {"count", "minimum", "maximum"} <= rl_fields
    assert {"count", "minimum", "maximum"}.isdisjoint(gate_fields)

    # The SHARED names are stated too, so the denominator is the whole field set
    # rather than a hand-picked disjoint subset: the divergence lives in the
    # count-and-extremes vocabulary, not in mean/std.
    assert {"mean", "std"} <= gate_fields
    assert {"mean", "std"} <= rl_fields


def test_the_gate_plane_admits_the_impossible_records_the_rl_plane_refuses() -> None:
    """WHAT IS CLAIMED: the gate-plane record is permissive BY DESIGN so that
    ``RewardScaleSanityGate``'s own MUST_FIRE fixtures can be constructed. That
    gate adjudicates a caller-supplied aggregate it never computed, so its job
    is to catch summaries no sample set could produce; delegating the record
    type to the validating RL class would make those fixtures unconstructible,
    disarming the detector at fixture-construction time rather than at gate
    time. This is #222's shape: the obvious unification is the wrong repair.
    WHAT IS NOT CLAIMED: that the gate ACCEPTS such a record as healthy. It
    does not -- it fails it, one layer up, under the objective-gate suite.
    """
    negative = GateRewardStats(n=-3, mean=0.5, std=0.3, min=-1.0, max=2.0)
    assert negative.n == -3

    # Fewer than two samples spanning a range is impossible as a summary, yet
    # must CONSTRUCT: the gate is what fails it, not the value object.
    single_with_spread = GateRewardStats(n=1, mean=0.0, std=0.0, min=-1.0, max=2.0)
    assert single_with_spread.n == 1
    assert single_with_spread.min < single_with_spread.max

    # The count=0 arm restates test_reward_stats_zero_count_refused_at_construction
    # on purpose: what is under test here is the CONTRAST, and a contrast that
    # cites only one of its two sides is not one. Deleting either arm as a
    # duplicate removes the comparison, not a redundancy.
    with pytest.raises(AdvantageRefusal) as zero_exc_info:
        RewardStats(count=0, mean=0.5, std=0.3, minimum=-1.0, maximum=2.0)
    assert "count=0" in str(zero_exc_info.value)

    with pytest.raises(AdvantageRefusal) as negative_exc_info:
        RewardStats(count=-3, mean=0.5, std=0.3, minimum=-1.0, maximum=2.0)
    assert "count=-3" in str(negative_exc_info.value)
