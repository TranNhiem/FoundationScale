"""Opt-in format-failure scoring for the Megatron online lane."""

from __future__ import annotations

from foundationscale.rl.megatron.online import (
    OnlineRow,
    format_failure_rows,
    overlong_shaped_rows,
    scored_group_advantages,
)


def _row(
    reward: float | None, *, gold_present: bool = True, finished: bool = True, length: int = 3
) -> OnlineRow:
    return OnlineRow(
        prompt_ids=(1, 2),
        completion_ids=tuple(range(length)),
        reward=reward,
        group=0,
        finished=finished,
        gold_present=gold_present,
    )


def test_abstention_with_gold_scores_the_declared_reward() -> None:
    rows = [_row(1.0), _row(None), _row(0.0)]
    out, converted = format_failure_rows(rows, reward=0.0)
    assert [r.reward for r in out] == [1.0, 0.0, 0.0]
    assert converted == 1


def test_abstention_without_gold_stays_unmeasured() -> None:
    out, converted = format_failure_rows([_row(None, gold_present=False)], reward=0.0)
    assert out[0].reward is None
    assert converted == 0


def test_input_rows_are_not_mutated() -> None:
    rows = [_row(None)]
    out, _ = format_failure_rows(rows, reward=-0.5)
    assert rows[0].reward is None
    assert out[0] is not rows[0]
    assert out[0].reward == -0.5


def test_converted_row_now_carries_a_negative_advantage() -> None:
    # Masked, the unparsable row is cheaper than a wrong one; scored, it is pushed down.
    rows = [_row(1.0), _row(1.0), _row(None), _row(None)]
    _, mask_before = scored_group_advantages(rows)
    assert mask_before[2] == 0
    out, _ = format_failure_rows(rows, reward=0.0)
    adv, mask = scored_group_advantages(out)
    assert mask[2] > 0 and adv[2] < 0


def test_overlong_shaping_still_applies_after_conversion() -> None:
    out, _ = format_failure_rows([_row(None, length=10)], reward=0.0)
    shaped, mean = overlong_shaped_rows(out, max_new_tokens=10, cache_tokens=4)
    assert shaped[0].reward == -1.0
    assert mean == -1.0


def test_default_row_declares_gold_present() -> None:
    row = OnlineRow(prompt_ids=(1,), completion_ids=(2,), reward=None, group=0, finished=True)
    assert row.gold_present is True
