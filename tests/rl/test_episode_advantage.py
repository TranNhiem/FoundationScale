"""ARL-17: episode-level baselines for compacted episodes (segments share one reward)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from foundationscale.rl.advantage import (
    SEGMENT_SEPARATOR,
    AdvantageRefusal,
    SessionGroupAdvantage,
    episode_id_of,
)
from foundationscale.rl.trainer import RLTrainer

MASK = ((1, 1),)


def _weights(result: Any) -> dict[int, float]:
    return {
        row: float(weights[0]) for row, weights in zip(result.rows, result.weights, strict=True)
    }


def _compute(
    rewards: Sequence[float | None],
    prompt_ids: Sequence[str] | None = None,
    episode_ids: Sequence[str] | None = None,
    **kwargs: Any,
) -> Any:
    return SessionGroupAdvantage(**kwargs).compute(
        prompt_ids=prompt_ids or ["p"] * len(rewards),
        rewards=rewards,
        mask=MASK * len(rewards),
        episode_ids=episode_ids,
    )


def test_without_episode_ids_nothing_changes() -> None:
    rewards = [1.0, 0.0, 0.0, 1.0]
    assert _weights(_compute(rewards)) == _weights(_compute(rewards, episode_ids=None))


def test_one_row_per_episode_equals_the_per_session_result() -> None:
    rewards = [1.0, 0.0, 0.0, 1.0]
    plain = _compute(rewards)
    own = _compute(rewards, episode_ids=["a", "b", "c", "d"])
    assert _weights(plain) == _weights(own)
    assert (plain.used, plain.offered) == (own.used, own.offered)


def test_each_episode_counts_once_and_its_advantage_reaches_every_segment() -> None:
    # Episode A compacted twice (3 segment rows, reward 1); B and C were not (reward 0).
    rewards = [1.0, 1.0, 1.0, 0.0, 0.0]
    episodes = ["A", "A", "A", "B", "C"]
    weights = _weights(_compute(rewards, episode_ids=episodes))
    # Baseline over 3 episodes: mean = 1/3, not 3/5 as 5 independent rows would give.
    assert weights[0] == weights[1] == weights[2] == pytest.approx(2 / 3)
    assert weights[3] == weights[4] == pytest.approx(-1 / 3)
    per_row = _weights(_compute(rewards))
    assert per_row[0] == pytest.approx(0.4)  # the bias the episode grouping removes


def test_min_group_size_counts_episodes_not_segments() -> None:
    # One episode split into 3 segments is still ONE member: its group has no baseline and
    # leaves the result, while a real 2-episode group next to it is kept.
    result = _compute(
        [1.0, 1.0, 1.0, 1.0, 0.0],
        prompt_ids=["p", "p", "p", "q", "q"],
        episode_ids=["A", "A", "A", "B", "C"],
    )
    assert set(_weights(result)) == {3, 4}
    with pytest.raises(AdvantageRefusal, match="nothing used"):
        _compute([1.0, 1.0, 1.0], episode_ids=["A", "A", "A"])


def test_an_abstained_episode_abstains_on_every_segment() -> None:
    rewards = [None, None, 1.0, 0.0]
    weights = _weights(_compute(rewards, episode_ids=["A", "A", "B", "C"]))
    assert set(weights) == {2, 3}


def test_refusals() -> None:
    with pytest.raises(AdvantageRefusal, match="differing rewards"):
        _compute([1.0, 0.0, 1.0], episode_ids=["A", "A", "B"])
    with pytest.raises(AdvantageRefusal, match="two prompt groups"):
        _compute(
            [1.0, 0.0, 1.0, 0.0], prompt_ids=["p", "p", "q", "q"], episode_ids=["A", "B", "A", "C"]
        )
    with pytest.raises(AdvantageRefusal, match="every row needs its episode id"):
        _compute([1.0, 0.0], episode_ids=["A"])
    with pytest.raises(AdvantageRefusal, match="non-empty str"):
        _compute([1.0, 0.0], episode_ids=["A", ""])


def test_episode_id_of() -> None:
    assert episode_id_of(f"t#0{SEGMENT_SEPARATOR}2") == "t#0"
    assert episode_id_of("t#0") == "t#0"
    assert episode_id_of(f"t#0{SEGMENT_SEPARATOR}x") == f"t#0{SEGMENT_SEPARATOR}x"
    assert episode_id_of(f"{SEGMENT_SEPARATOR}1") == f"{SEGMENT_SEPARATOR}1"


class _Batch:
    def __init__(self, columns: Mapping[str, Sequence[Any]]) -> None:
        self.columns = dict(columns)

    def column(self, name: str) -> Sequence[Any]:
        return self.columns[name]


def test_trainer_derives_episode_ids_only_when_segments_are_present() -> None:
    plain = _Batch({"session_id": ["t#0", "t#1"]})
    assert RLTrainer._episode_ids_from_batch(plain, kept_row_indices=[0, 1]) is None  # type: ignore[arg-type]
    assert RLTrainer._episode_ids_from_batch(_Batch({}), kept_row_indices=[]) is None  # type: ignore[arg-type]
    segmented = _Batch({"session_id": ["t#0::seg0", "t#0::seg1", "t#1"]})
    assert RLTrainer._episode_ids_from_batch(segmented, kept_row_indices=[1, 2]) == ["t#0", "t#1"]  # type: ignore[arg-type]
