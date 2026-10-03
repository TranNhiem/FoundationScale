"""Soft overlong punishment for the Megatron online lane (DAPO-style length shaping)."""

from __future__ import annotations

import pytest

from foundationscale.rl.megatron.online import OnlineRow, overlong_shaped_rows


def _row(
    length: int,
    *,
    reward: float | None = 1.0,
    finished: bool = True,
    group: int = 0,
) -> OnlineRow:
    """A rollout row with ``length`` completion tokens; ids are generated, never re-derived."""

    return OnlineRow(
        prompt_ids=(1, 2),
        completion_ids=tuple(range(length)),
        reward=reward,
        group=group,
        finished=finished,
    )


def test_no_penalty_at_or_below_start() -> None:
    rows = [_row(6), _row(2), _row(0)]
    out, mean = overlong_shaped_rows(rows, max_new_tokens=10, cache_tokens=4)
    assert [r.reward for r in out] == [1.0, 1.0, 1.0]
    assert mean == 0.0


def test_linear_penalty_inside_cache_zone_with_exact_values() -> None:
    rows = [_row(7), _row(8), _row(9)]
    out, mean = overlong_shaped_rows(rows, max_new_tokens=10, cache_tokens=4)
    # start = 10 - 4 = 6: -(L - 6) / 4 * 1.0 gives -0.25, -0.5, -0.75 exactly.
    assert [r.reward for r in out] == [0.75, 0.5, 0.25]
    assert out[1].reward == 0.5  # L == 8 -> -0.5 exactly, the ramp's midpoint
    assert mean == -0.5


def test_full_penalty_at_budget_or_when_unfinished() -> None:
    rows = [_row(10), _row(14), _row(6, finished=False), _row(0, finished=False)]
    out, mean = overlong_shaped_rows(rows, max_new_tokens=10, cache_tokens=4, factor=2.0)
    assert [r.reward for r in out] == [-1.0, -1.0, -1.0, -1.0]
    assert mean == -2.0


def test_factor_scales_the_whole_shape() -> None:
    rows = [_row(8), _row(10)]
    out, mean = overlong_shaped_rows(rows, max_new_tokens=10, cache_tokens=4, factor=2.0)
    assert [r.reward for r in out] == [0.0, -1.0]
    assert mean == -1.5


def test_abstention_stays_none_and_is_excluded_from_mean() -> None:
    rows = [_row(8, reward=None), _row(0, reward=None, finished=False), _row(8)]
    out, mean = overlong_shaped_rows(rows, max_new_tokens=10, cache_tokens=4)
    assert out[0].reward is None
    assert out[1].reward is None
    assert out[2].reward == 0.5
    assert mean == -0.5  # only the MEASURED row's penalty: abstention borrows no weight


def test_mean_is_mean_over_scored_rows_only() -> None:
    rows = [_row(8), _row(6), _row(7), _row(3, reward=None)]
    out, mean = overlong_shaped_rows(rows, max_new_tokens=10, cache_tokens=4)
    assert [r.reward for r in out if r.reward is not None] == [0.5, 1.0, 0.75]
    assert mean == -0.25  # (-0.5 + 0.0 + -0.25) / 3


def test_mean_is_zero_when_every_row_stayed_unmeasured() -> None:
    rows = [_row(8, reward=None), _row(1, reward=None)]
    out, mean = overlong_shaped_rows(rows, max_new_tokens=10, cache_tokens=4)
    assert [r.reward for r in out] == [None, None]
    assert mean == 0.0


@pytest.mark.parametrize(
    ("max_new_tokens", "cache_tokens", "factor", "match"),
    [
        (0, 1, 1.0, r"overlong_shaped_rows: max_new_tokens must be >= 1"),
        (10, 0, 1.0, r"overlong_shaped_rows: cache_tokens must be >= 1"),
        (10, -3, 1.0, r"overlong_shaped_rows: cache_tokens must be >= 1"),
        (4, 8, 1.0, r"cache_tokens \(8\) must be <= max_new_tokens \(4\)"),
        (10, 4, -0.5, r"overlong_shaped_rows: factor must be >= 0"),
    ],
)
def test_invalid_arguments_raise(
    max_new_tokens: int,
    cache_tokens: int,
    factor: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        overlong_shaped_rows(
            [_row(6)],
            max_new_tokens=max_new_tokens,
            cache_tokens=cache_tokens,
            factor=factor,
        )


def test_input_rows_are_unmodified_and_new_objects_are_returned() -> None:
    original = [_row(8), _row(12, reward=None)]
    out, _ = overlong_shaped_rows(original, max_new_tokens=10, cache_tokens=4)
    assert out[0] is not original[0]
    assert out[1] is not original[1]
    assert original[0].reward == 1.0
    assert original[1].reward is None
    assert original[0].completion_ids == tuple(range(8))
    assert original[1].completion_ids == tuple(range(12))
    assert out[0].completion_ids == original[0].completion_ids


def test_empty_rows_return_empty_and_zero_mean() -> None:
    out, mean = overlong_shaped_rows([], max_new_tokens=10, cache_tokens=4)
    assert out == []
    assert mean == 0.0
