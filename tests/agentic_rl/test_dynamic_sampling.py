"""ARL-G06 dynamic sampling (DAPO-style refill) in RolloutHost."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foundationscale.agentic_rl.rewards.base import RewardVerdict
from foundationscale.agentic_rl.rollout_host import RolloutHostRefusal

try:
    import test_engine_recovery as fakes  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - package-style collection
    from tests.agentic_rl import test_engine_recovery as fakes  # type: ignore[no-redef]


class ScriptedReward:
    """The k-th score call for a uid returns ``table[uid][k % len]`` (a group's set of values
    is then fixed whatever order its replicas finish in); uids absent from the table abstain."""

    def __init__(self, table: dict[str, tuple[float | None, ...]]) -> None:
        self.table = table
        self.calls: dict[str, int] = {}

    async def score(self, task: Any, outcome: Any) -> RewardVerdict:
        k = self.calls.get(task.uid, 0)
        self.calls[task.uid] = k + 1
        values = self.table.get(task.uid)
        value = None if values is None else values[k % len(values)]
        return RewardVerdict(value, None if value is not None else "scripted_abstain")


UIDS = [f"task{i}" for i in range(8)]


def _host(tmp_path: Path, table: dict[str, tuple[float | None, ...]], **kw: Any) -> Any:
    return fakes._host(
        tmp_path,
        uids=UIDS,
        client_factory=lambda index: fakes.SubmitClient("x"),
        reward=ScriptedReward(table),
        tasks_per_step=2,
        gates=False,
        **kw,
    )


def _uids(batch: Any) -> list[str]:
    return sorted(set(batch.columns["uid"]))


def test_off_keeps_zero_spread_groups(tmp_path: Path) -> None:
    table = {u: (1.0, 1.0) for u in UIDS}  # every group saturated
    batch = _host(tmp_path, table).rollout(0)
    assert len(_uids(batch)) == 2


def test_refill_replaces_zero_spread_groups_with_informative_ones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # saturated everywhere except two tasks; refill must find them
    table = {u: (1.0, 1.0) for u in UIDS}
    host = _host(tmp_path, table, dynamic_sampling="refill", max_refill_rounds=3)
    pool = host.tasks.batch(0, 8)
    informative = [pool[3].uid, pool[6].uid]  # only reachable through refill rounds
    for uid in informative:
        table[uid] = (1.0, 0.0)
    batch = host.rollout(0)
    assert _uids(batch) == sorted(informative)
    err = capsys.readouterr().err
    line = next(ln for ln in err.splitlines() if ln.startswith("[agentic-rl:dynamic-sampling]"))
    stats = json.loads(line.split(" ", 1)[1])
    assert stats["groups_kept"] == 2 and stats["rounds"] == 3
    # the summary line still describes round 0 only (two saturated groups, mean 1.0)
    summary = next(ln for ln in err.splitlines() if ln.startswith("[agentic-rl:rollout]"))
    assert json.loads(summary.split(" ", 1)[1])["reward_mean"] == 1.0


def test_refill_never_drops_a_group_with_an_abstention(tmp_path: Path) -> None:
    table: dict[str, tuple[float | None, ...]] = {u: (1.0, 1.0) for u in UIDS}
    host = _host(tmp_path, table, dynamic_sampling="refill", max_refill_rounds=1)
    pool = host.tasks.batch(0, 4)
    table[pool[0].uid] = (None, None)  # infra-like: abstained rows must reach the gate
    batch = host.rollout(0)
    assert pool[0].uid in _uids(batch)


def test_refill_with_no_signal_anywhere_returns_round_zero(tmp_path: Path) -> None:
    table = {u: (0.0, 0.0) for u in UIDS}
    host = _host(tmp_path, table, dynamic_sampling="refill", max_refill_rounds=2)
    batch = host.rollout(0)
    assert _uids(batch) == sorted(t.uid for t in host.tasks.batch(0, 6)[:2])


def test_bad_dynamic_sampling_settings_are_refused(tmp_path: Path) -> None:
    with pytest.raises(RolloutHostRefusal, match="dynamic_sampling"):
        _host(tmp_path, {}, dynamic_sampling="filter")
    with pytest.raises(RolloutHostRefusal, match="max_refill_rounds"):
        _host(tmp_path, {}, dynamic_sampling="refill", max_refill_rounds=0)


def _build(**rollout: Any) -> Any:
    import importlib.util

    from foundationscale.agentic_rl.config import build_config

    spec = importlib.util.spec_from_file_location(
        "_test_config_raw_ds", Path(__file__).with_name("test_config.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = module._raw_config()
    raw["rollout"] = {**raw["rollout"], **rollout}
    return build_config(raw, {})


def test_config_keys_default_accept_and_refuse() -> None:
    from foundationscale.agentic_rl.config import AgenticConfigRefusal

    default = _build().rollout
    assert (default.dynamic_sampling, default.max_refill_rounds) == ("off", 2)
    chosen = _build(dynamic_sampling="refill", max_refill_rounds=4).rollout
    assert (chosen.dynamic_sampling, chosen.max_refill_rounds) == ("refill", 4)
    with pytest.raises(AgenticConfigRefusal, match="dynamic_sampling"):
        _build(dynamic_sampling="dapo")
    with pytest.raises(AgenticConfigRefusal, match="max_refill_rounds"):
        _build(max_refill_rounds=0)
