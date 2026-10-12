"""FS music reward and tasks in verl's shapes (P0.3 parity)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from foundationscale.agentic_rl import verl_rewards as vr

INFO = {"abc2midi_bin": "/bin/abc2midi", "timeout_s": 30.0}


class _FakeReward:
    def __init__(self, value: float | None, reason: str | None) -> None:
        self.result = SimpleNamespace(value=value, reason=reason)
        self.seen: list[str] = []

    def score(self, response: str) -> SimpleNamespace:
        self.seen.append(response)
        return self.result


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch):
    def install(value: float | None, reason: str | None = None) -> _FakeReward:
        reward = _FakeReward(value, reason)
        monkeypatch.setattr(vr, "_music_reward", lambda binary, timeout: reward)
        return reward

    return install


def test_score_returns_the_fs_value(fake) -> None:
    """A measured FS score passes through as a float."""
    reward = fake(0.62)
    assert vr.music_compute_score(vr.DATA_SOURCE_MUSIC, "X:1\nK:C\nC", "", INFO) == 0.62
    assert reward.seen == ["X:1\nK:C\nC"]


def test_earned_zero_is_a_score_not_an_abstention(fake) -> None:
    """A model-attributable zero (no_abc) is returned as 0.0."""
    fake(0.0, "no_abc")
    assert vr.music_compute_score(vr.DATA_SOURCE_MUSIC, "no music", "", INFO) == 0.0


def test_infra_abstention_raises_instead_of_scoring(fake) -> None:
    """An FS abstention stops the run rather than becoming a fabricated number."""
    fake(None, "infra:abc2midi_timeout")
    with pytest.raises(vr.VerlRewardAbstention, match="infra:abc2midi_timeout"):
        vr.music_compute_score(vr.DATA_SOURCE_MUSIC, "X:1", "", INFO)


@pytest.mark.parametrize(
    ("source", "info", "match"),
    [
        ("openai/gsm8k", INFO, "data_source"),
        (vr.DATA_SOURCE_MUSIC, None, "abc2midi_bin"),
        (vr.DATA_SOURCE_MUSIC, {"abc2midi_bin": ""}, "abc2midi_bin"),
        (vr.DATA_SOURCE_MUSIC, {"abc2midi_bin": "x", "timeout_s": True}, "timeout_s"),
        (vr.DATA_SOURCE_MUSIC, {"abc2midi_bin": "x", "timeout_s": "60"}, "timeout_s"),
    ],
)
def test_bad_inputs_are_refused(fake, source, info, match) -> None:
    """Wrong data source or undeclared scorer settings are refused by name."""
    fake(0.5)
    with pytest.raises(vr.VerlTaskRefusal, match=match):
        vr.music_compute_score(source, "X:1", "", info)


def test_real_music_reward_is_built_from_declared_settings() -> None:
    """The cached constructor builds the real FS MusicReward with the row's settings."""
    reward = vr._music_reward("/opt/abc2midi", 12.0)
    assert (reward.abc2midi_bin, reward.timeout_s) == ("/opt/abc2midi", 12.0)
    assert vr._music_reward("/opt/abc2midi", 12.0) is reward


def test_rows_from_jsonl_converts_each_task() -> None:
    """Each FS task becomes one verl row carrying the declared scorer settings."""
    lines = [
        json.dumps({"uid": "a", "messages": [{"role": "user", "content": "compose"}]}),
        "",
        json.dumps({"uid": "b", "messages": [{"role": "user", "content": "again"}], "x": 1}),
    ]
    rows = vr.music_rows_from_jsonl(lines, abc2midi_bin="/bin/abc2midi", timeout_s=45)
    assert [r["extra_info"]["uid"] for r in rows] == ["a", "b"]
    assert [r["extra_info"]["index"] for r in rows] == [0, 1]
    assert rows[0] == {
        "data_source": "fs_music",
        "prompt": [{"role": "user", "content": "compose"}],
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"uid": "a", "index": 0, "abc2midi_bin": "/bin/abc2midi", "timeout_s": 45.0},
    }


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ("{not json", "not JSON"),
        (json.dumps({"messages": [{"role": "user", "content": "x"}]}), "uid"),
        (json.dumps({"uid": "a", "messages": []}), "messages"),
        (json.dumps({"uid": "a", "messages": [{"role": "user"}]}), "messages"),
        (json.dumps(["not", "a", "dict"]), "uid"),
    ],
)
def test_bad_rows_are_refused(line: str, match: str) -> None:
    """Malformed task rows are refused with the line number."""
    with pytest.raises(vr.VerlTaskRefusal, match=match):
        vr.music_rows_from_jsonl([line], abc2midi_bin="x")


def test_empty_task_file_is_refused() -> None:
    """Zero rows is refused, never an empty dataset."""
    with pytest.raises(vr.VerlTaskRefusal, match="0 task rows"):
        vr.music_rows_from_jsonl(["", "  "], abc2midi_bin="x")
