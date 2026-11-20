"""ARL-G13: unbiased pass@k and the paired task bootstrap."""

from __future__ import annotations

import math

import pytest

from foundationscale.agentic_rl.eval_stats import EvalStatsRefusal, paired_bootstrap, pass_at_k


def test_pass_at_k_matches_the_unbiased_estimator_by_hand() -> None:
    per_task = {"t1": [1.0, 0.0, 0.0, 0.0], "t2": [1.0, 1.0, 0.0, 0.0]}
    assert pass_at_k(per_task, 1) == pytest.approx((0.25 + 0.5) / 2)
    # k=2: t1 = 1 - C(3,2)/C(4,2) = 0.5 ; t2 = 1 - C(2,2)/C(4,2) = 5/6
    assert pass_at_k(per_task, 2) == pytest.approx((0.5 + 5 / 6) / 2)
    assert pass_at_k(per_task, 4) == pytest.approx(1.0)


def test_pass_at_k_drops_abstentions_and_refuses_too_few_samples() -> None:
    assert pass_at_k({"t": [1.0, None, 0.0]}, 1) == pytest.approx(0.5)
    with pytest.raises(EvalStatsRefusal, match="< k=3"):
        pass_at_k({"t": [1.0, None, 0.0]}, 3)
    with pytest.raises(EvalStatsRefusal, match="k=0"):
        pass_at_k({"t": [1.0]}, 0)
    with pytest.raises(EvalStatsRefusal, match="finite"):
        pass_at_k({"t": [math.nan]}, 1)
    with pytest.raises(EvalStatsRefusal, match="finite"):
        pass_at_k({"t": [True]}, 1)  # a bool is not a score


def test_identical_evaluations_have_zero_difference_inside_the_ci() -> None:
    a = {f"t{i}": [float(i % 2), float(i % 3 == 0)] for i in range(30)}
    result = paired_bootstrap(a, a, resamples=500)
    assert result.diff == 0.0 and result.ci_low == 0.0 and result.ci_high == 0.0
    assert not result.excludes_zero and result.p_le_zero == 1.0


def test_a_uniform_shift_is_recovered_exactly_and_excludes_zero() -> None:
    a = {f"t{i}": [0.25] for i in range(20)}
    b = {f"t{i}": [0.75] for i in range(20)}
    result = paired_bootstrap(a, b, resamples=500)
    assert result.diff == pytest.approx(0.5)
    assert (result.ci_low, result.ci_high) == (pytest.approx(0.5), pytest.approx(0.5))
    assert result.excludes_zero and result.p_le_zero == 0.0


def test_a_noisy_small_difference_is_not_called_a_finding() -> None:
    a = {f"t{i}": [1.0 if i % 2 else 0.0] for i in range(30)}
    b = dict(a)
    b["t0"] = [1.0]  # one task flipped: +1/30
    result = paired_bootstrap(a, b, resamples=2000)
    assert result.diff == pytest.approx(1 / 30)
    assert result.ci_low <= 0.0 <= result.ci_high


def test_paired_bootstrap_refusals_and_drops() -> None:
    with pytest.raises(EvalStatsRefusal, match="task sets differ"):
        paired_bootstrap({"t1": [1.0], "t2": [0.0]}, {"t1": [1.0], "t3": [0.0]})
    with pytest.raises(EvalStatsRefusal, match="need >= 2"):
        paired_bootstrap({"t1": [1.0], "t2": [None]}, {"t1": [0.0], "t2": [1.0]})
    result = paired_bootstrap(
        {"t1": [1.0], "t2": [None], "t3": [0.0]},
        {"t1": [0.0], "t2": [1.0], "t3": [0.0]},
        resamples=200,
    )
    assert (result.n_tasks, result.n_dropped) == (2, 1)
    with pytest.raises(EvalStatsRefusal, match="resamples"):
        paired_bootstrap({"t1": [1.0], "t2": [0.0]}, {"t1": [1.0], "t2": [0.0]}, resamples=10)
