"""Session-group advantage: the one estimator whose reward may abstain.

WHAT IS CLAIMED: every ``partial_group_policy`` branch, the None exclusion,
``min_valid``, ``min_group_size``, the zero-variance drop, optional
population-std normalisation, every name-bearing refusal, and the exact
``used``/``offered``/``rows`` account -- including that harness-suffixed
grouping keys form SEPARATE baselines rather than one pooled one.

WHAT IS NOT CLAIMED: that the grouping key is an episode beyond what the
caller passed in; the multi-output session seam is a later sibling protocol.
"""

from __future__ import annotations

import math

import pytest

from foundationscale.rl.advantage import (
    AdvantageConfigRefusal,
    AdvantageFn,
    AdvantageRefusal,
    SessionGroupAdvantage,
)


def test_the_estimator_satisfies_the_advantage_fn_protocol() -> None:
    # Structural conformance -- the protocol has one keyword-only compute --
    # and the estimator is constructible at its declared defaults.
    estimator = SessionGroupAdvantage()
    assert isinstance(estimator, AdvantageFn)
    assert estimator.name == "session_group"


def test_centred_baseline_over_sessions_with_a_zero_variance_drop() -> None:
    # s1 rewards (1, 3) -> mean 2 -> -1, +1. s2 rewards (2, 2) -> zero
    # spread -> DROPPED, so its rows leave the result entirely rather than
    # being written as measured zeros.
    estimator = SessionGroupAdvantage(min_group_size=2)
    result = estimator.compute(
        prompt_ids=("s1", "s1", "s2", "s2"),
        rewards=(1.0, 3.0, 2.0, 2.0),
        mask=((1, 1), (1, 1), (1, 1), (1, 1)),
    )
    assert result.weights == ((-1.0, -1.0), (1.0, 1.0))
    assert result.rows == (0, 1)
    assert result.used == 2
    assert result.offered == 4
    assert result.rewards.count == 2
    assert result.rewards.mean == pytest.approx(2.0)
    assert result.rewards.std == pytest.approx(1.0)
    assert result.method == "session_group"


def test_std_normalisation_divides_by_the_population_std() -> None:
    # rewards (0, 2, 6): mean 8/3, population variance over 3 members.
    estimator = SessionGroupAdvantage(min_group_size=2, normalise_by_std=True)
    result = estimator.compute(
        prompt_ids=("s", "s", "s"),
        rewards=(0.0, 2.0, 6.0),
        mask=((1,), (1,), (1,)),
    )
    mean = 8.0 / 3.0
    population_std = math.sqrt(((0.0 - mean) ** 2 + (2.0 - mean) ** 2 + (6.0 - mean) ** 2) / 3.0)
    assert result.used == 3
    for row, reward in enumerate((0.0, 2.0, 6.0)):
        assert result.weights[row][0] == pytest.approx((reward - mean) / population_std)
    assert result.rewards.std == pytest.approx(population_std)


def test_one_is_not_true_for_normalise_by_std() -> None:
    # The policy flag is a real bool: a truthy stand-in would hide which of
    # the two behaviours a manifest recorded as chosen.
    for bad in (1, 0, "true", 2.0):
        with pytest.raises(AdvantageConfigRefusal, match="must be a bool"):
            SessionGroupAdvantage(normalise_by_std=bad)  # type: ignore[arg-type]


def test_none_reward_excludes_the_row_and_never_scores_it_zero() -> None:
    # valid rows 0, 2, 3 -> rewards 1, 3, 5 -> mean 3 -> -2, 0, +2. Row 1
    # abstained and is EXCLUDED: no weight row, and `rows` says so.
    estimator = SessionGroupAdvantage(min_group_size=2, min_valid=2)
    result = estimator.compute(
        prompt_ids=("s", "s", "s", "s"),
        rewards=(1.0, None, 3.0, 5.0),
        mask=((1, 1), (1, 1), (1, 1), (1, 1)),
    )
    assert result.rows == (0, 2, 3)
    assert result.used == 3
    assert result.offered == 4
    assert result.weights == ((-2.0, -2.0), (0.0, 0.0), (2.0, 2.0))
    assert result.rewards.count == 3
    assert result.rewards.mean == pytest.approx(3.0)


def test_refuse_policy_names_the_group_and_the_row() -> None:
    estimator = SessionGroupAdvantage(min_group_size=2, partial_group_policy="refuse")
    with pytest.raises(AdvantageRefusal) as exc_info:
        estimator.compute(
            prompt_ids=("s1", "s1", "s2"),
            rewards=(1.0, None, 3.0),
            mask=((1,), (1,), (1,)),
        )
    message = str(exc_info.value)
    assert "row 1" in message
    assert "'s1'" in message
    assert "1 of 3" in message
    assert "partial_group_policy='refuse'" in message


def test_drop_group_drops_a_partially_abstained_group_wholesale() -> None:
    # s1 holds an abstention -> contributes NO rows; s2 is kept whole.
    estimator = SessionGroupAdvantage(min_group_size=2, partial_group_policy="drop_group")
    result = estimator.compute(
        prompt_ids=("s1", "s1", "s2", "s2"),
        rewards=(1.0, None, 2.0, 4.0),
        mask=((1,), (1,), (1,), (1,)),
    )
    assert result.rows == (2, 3)
    assert result.weights == ((-1.0,), (1.0,))
    assert result.used == 2
    assert result.offered == 4


def test_shrink_keeps_a_partially_abstained_group_at_min_valid() -> None:
    # s1 keeps rows 0, 2 -> rewards 1, 3 -> mean 2 -> -1, +1. s2 keeps both
    # -> rewards 2, 6 -> mean 4 -> -2, +2.
    estimator = SessionGroupAdvantage(
        min_group_size=2,
        min_valid=2,
        partial_group_policy="shrink",
    )
    result = estimator.compute(
        prompt_ids=("s1", "s1", "s1", "s2", "s2"),
        rewards=(1.0, None, 3.0, 2.0, 6.0),
        mask=((1,), (1,), (1,), (1,), (1,)),
    )
    assert result.rows == (0, 2, 3, 4)
    assert result.weights == ((-1.0,), (1.0,), (-2.0,), (2.0,))
    assert result.used == 4
    assert result.offered == 5


def test_shrink_drops_a_group_below_min_valid() -> None:
    # s1 kept 1 valid member, short of min_valid=2 -> dropped.
    estimator = SessionGroupAdvantage(
        min_group_size=2,
        min_valid=2,
        partial_group_policy="shrink",
    )
    result = estimator.compute(
        prompt_ids=("s1", "s1", "s2", "s2"),
        rewards=(1.0, None, 2.0, 6.0),
        mask=((1,), (1,), (1,), (1,)),
    )
    assert result.rows == (2, 3)
    assert result.weights == ((-2.0,), (2.0,))
    assert result.offered == 4


def test_min_group_size_drops_a_small_offered_group() -> None:
    # group "a" was OFFERED 2 rows against a minimum of 3 -> dropped,
    # exactly as GroupNormalisedAdvantage drops it.
    estimator = SessionGroupAdvantage(min_group_size=3)
    result = estimator.compute(
        prompt_ids=("a", "a", "b", "b", "b"),
        rewards=(1.0, 3.0, 1.0, 2.0, 3.0),
        mask=((1,), (1,), (1,), (1,), (1,)),
    )
    assert result.rows == (2, 3, 4)
    assert result.weights == ((-1.0,), (0.0,), (1.0,))
    assert result.used == 3
    assert result.offered == 5


def test_zero_variance_group_is_dropped_not_zeroed() -> None:
    # The deliberate divergence from CentredAdvantage, stated in the class
    # docstring: zero spread means no relative signal, so the group EXITS
    # rather than filling the denominator with measured zeros.
    estimator = SessionGroupAdvantage(min_group_size=2)
    result = estimator.compute(
        prompt_ids=("flat", "flat", "live", "live"),
        rewards=(1.0, 1.0, 1.0, 3.0),
        mask=((1,), (1,), (1,), (1,)),
    )
    assert result.rows == (2, 3)
    assert result.weights == ((-1.0,), (1.0,))
    assert result.offered == 4
    assert result.used == 2


def test_every_row_excluded_fails_closed() -> None:
    estimator = SessionGroupAdvantage(min_group_size=2)
    with pytest.raises(AdvantageRefusal, match="nothing used means nothing measured"):
        estimator.compute(
            prompt_ids=("s", "s"),
            rewards=(1.0, 1.0),
            mask=((1,), (1,)),
        )


def test_masked_positions_carry_a_literal_zero() -> None:
    # "drop" is zero-variance and excluded; s1 keeps rows 0, 1 with -1/+1
    # broadcast only over the unmasked positions.
    estimator = SessionGroupAdvantage(min_group_size=2)
    result = estimator.compute(
        prompt_ids=("s1", "s1", "drop", "drop"),
        rewards=(1.0, 3.0, 1.0, 1.0),
        mask=((1, 0), (0, 1), (1,), (1,)),
    )
    assert result.weights == ((-1.0, 0.0), (0.0, 1.0))
    assert result.rows == (0, 1)
    assert result.offered == 4
    assert result.used == 2


def test_harness_suffixed_keys_form_separate_baselines() -> None:
    # "u::a" -> mean 2, std 1 -> -1, +1. "u::b" -> mean 7, std 2 ->
    # -2, +2. A pooled baseline over all four would read mean 4.5.
    estimator = SessionGroupAdvantage(min_group_size=2)
    result = estimator.compute(
        prompt_ids=("u::a", "u::a", "u::b", "u::b"),
        rewards=(1.0, 3.0, 5.0, 9.0),
        mask=((1,), (1,), (1,), (1,)),
    )
    assert result.weights == ((-1.0,), (1.0,), (-2.0,), (2.0,))
    assert result.used == 4
    assert result.offered == 4


def test_refuses_a_non_finite_reward_naming_the_row() -> None:
    estimator = SessionGroupAdvantage(min_group_size=2)
    with pytest.raises(AdvantageRefusal, match="which is not finite"):
        estimator.compute(
            prompt_ids=("s", "s"),
            rewards=(float("nan"), 1.0),
            mask=((1,), (1,)),
        )


def test_refuses_a_non_convertible_reward_naming_the_row() -> None:
    estimator = SessionGroupAdvantage(min_group_size=2)
    with pytest.raises(AdvantageRefusal, match="does not convert to a scalar float"):
        estimator.compute(
            prompt_ids=("s", "s"),
            rewards=("x", 1.0),
            mask=((1,), (1,)),
        )


def test_refuses_ragged_lengths_naming_all_three_counts() -> None:
    estimator = SessionGroupAdvantage()
    with pytest.raises(AdvantageRefusal) as exc_info:
        estimator.compute(
            prompt_ids=("s",),
            rewards=(1.0, None),
            mask=((1,), (1,)),
        )
    message = str(exc_info.value)
    assert "1 prompt ids" in message
    assert "2 rewards" in message
    assert "2 mask rows" in message


def test_refuses_a_fractional_mask_entry() -> None:
    estimator = SessionGroupAdvantage()
    with pytest.raises(AdvantageRefusal, match="supervision mask entries must be 0 or 1"):
        estimator.compute(
            prompt_ids=("s", "s"),
            rewards=(1.0, 2.0),
            mask=((1, 0.5), (1, 1)),
        )


def test_refuses_a_row_with_no_supervised_token() -> None:
    estimator = SessionGroupAdvantage()
    with pytest.raises(AdvantageRefusal, match="supervises 0 of 1 positions"):
        estimator.compute(
            prompt_ids=("s", "s"),
            rewards=(1.0, 2.0),
            mask=((0,), (1,)),
        )


def test_refuses_an_empty_batch() -> None:
    estimator = SessionGroupAdvantage()
    with pytest.raises(AdvantageRefusal, match="compute got 0 samples"):
        estimator.compute(prompt_ids=(), rewards=(), mask=())


def test_config_refuses_a_bad_min_group_size() -> None:
    with pytest.raises(AdvantageConfigRefusal, match="min_group_size=True"):
        SessionGroupAdvantage(min_group_size=True)
    with pytest.raises(AdvantageConfigRefusal, match="at least 2 samples"):
        SessionGroupAdvantage(min_group_size=1)


def test_config_refuses_min_valid_outside_its_declared_range() -> None:
    for bad in (1, 5, True, 2.5):
        with pytest.raises(AdvantageConfigRefusal, match="min_valid="):
            SessionGroupAdvantage(min_group_size=4, min_valid=bad)  # type: ignore[arg-type]


def test_config_refuses_an_undeclared_partial_group_policy() -> None:
    with pytest.raises(AdvantageConfigRefusal, match="partial_group_policy="):
        SessionGroupAdvantage(partial_group_policy="average")  # type: ignore[arg-type]


def test_config_refuses_an_empty_method_name() -> None:
    with pytest.raises(AdvantageConfigRefusal, match="absence of a name is not a name"):
        SessionGroupAdvantage(name="")


def test_config_accepts_a_real_bool_and_a_coherent_min_valid() -> None:
    # Positive control for the two refusals above: the admissible edge reads
    # as configured, not as "whatever passed".
    estimator = SessionGroupAdvantage(
        normalise_by_std=False,
        min_group_size=2,
        min_valid=2,
        partial_group_policy="shrink",
    )
    assert estimator.normalise_by_std is False
    assert estimator.min_valid == 2


def test_float_noise_never_reads_as_spread_in_an_equal_reward_group() -> None:
    # 0.1 * 3 sums to 0.30000000000000004: a computed std of ~1e-17 must not
    # survive into a full-strength +-1 under normalisation.
    for normalise in (False, True):
        estimator = SessionGroupAdvantage(min_group_size=2, normalise_by_std=normalise)
        with pytest.raises(AdvantageRefusal):
            estimator.compute(
                prompt_ids=("s", "s", "s"),
                rewards=(0.1, 0.1, 0.1),
                mask=((1,), (1,), (1,)),
            )


def test_an_infra_row_with_an_all_zero_mask_leaves_the_group_not_the_batch() -> None:
    estimator = SessionGroupAdvantage(min_group_size=2, partial_group_policy="shrink", min_valid=2)
    result = estimator.compute(
        prompt_ids=("s", "s", "s"),
        rewards=(None, 1.0, 3.0),
        mask=((0, 0, 0), (1, 1), (1, 1)),
    )
    assert result.rows == (1, 2)
    assert result.used == 2
    assert result.offered == 3
    assert [row[0] for row in result.weights] == [-1.0, 1.0]


def test_an_infra_row_with_an_empty_mask_leaves_the_group_not_the_batch() -> None:
    # A zero-turn INFRA trajectory (contracts.Trajectory with prompt_turns=
    # turns=()) has no token positions at all, not merely an all-0 mask: the
    # empty row must be admitted for an abstained (None) reward exactly as the
    # all-0 case already is.
    estimator = SessionGroupAdvantage(min_group_size=2, partial_group_policy="shrink", min_valid=2)
    result = estimator.compute(
        prompt_ids=("s", "s", "s"),
        rewards=(None, 1.0, 3.0),
        mask=((), (1, 1), (1, 1)),
    )
    assert result.rows == (1, 2)
    assert result.used == 2
    assert result.offered == 3
    assert [row[0] for row in result.weights] == [-1.0, 1.0]


def test_a_scored_row_that_supervises_nothing_is_still_refused() -> None:
    estimator = SessionGroupAdvantage(min_group_size=2)
    with pytest.raises(AdvantageRefusal, match="supervises 0 of 2"):
        estimator.compute(prompt_ids=("s", "s"), rewards=(0.0, 1.0), mask=((0, 0), (1, 1)))


def test_refuse_policy_counts_every_abstention_and_ignores_undersized_groups() -> None:
    estimator = SessionGroupAdvantage(min_group_size=3, partial_group_policy="refuse", min_valid=2)
    with pytest.raises(AdvantageRefusal, match="2 of 4 offered rewards"):
        estimator.compute(
            prompt_ids=("a", "a", "a", "b"),
            rewards=(None, None, 1.0, 2.0),
            mask=((1,), (1,), (1,), (1,)),
        )
    with pytest.raises(AdvantageRefusal):
        # Only an undersized group abstains: it is dropped, not refused, and
        # with nothing else usable the result fails closed as empty.
        estimator.compute(prompt_ids=("s", "s"), rewards=(1.0, None), mask=((1,), (1,)))
