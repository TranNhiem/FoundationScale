"""ARL-20 frontier curriculum: DifficultyRegistry, FrontierTaskSource, rollout wiring."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from foundationscale.agentic_rl.curriculum import DifficultyRegistry, FrontierTaskSource
from foundationscale.agentic_rl.tasks import TaskSource, TaskSourceRefusal

try:
    import test_engine_recovery as fakes  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - package-style collection
    from tests.agentic_rl import test_engine_recovery as fakes  # type: ignore[no-redef]


def _source(tmp_path: Path, n: int, **kw: Any) -> FrontierTaskSource:
    uids = [f"t{i}" for i in range(n)]
    return FrontierTaskSource(base=TaskSource(path=fakes._write_tasks(tmp_path, uids)), **kw)


def test_registry_uses_a_beta_posterior_clips_rewards_and_ignores_abstentions() -> None:
    reg = DifficultyRegistry()
    assert reg.pass_rate("new") == 0.5
    for r in (1.0, 0.0, 3.0, -2.0, None):  # 3.0 counts as 1, -2.0 as 0, None not at all
        reg.update("t", r)
    assert reg.counts["t"] == 4 and reg.sums["t"] == 2.0
    assert reg.pass_rate("t") == pytest.approx(3 / 6)
    with pytest.raises(TaskSourceRefusal):
        reg.update("t", True)  # a bool is not a reward
    with pytest.raises(TaskSourceRefusal):
        reg.update("t", float("nan"))


def test_saturated_tasks_sit_at_the_floor_and_unseen_tasks_at_full_weight(tmp_path: Path) -> None:
    src = _source(tmp_path, 3, min_obs=4, floor=0.05)
    for _ in range(4):
        src.registry.update("t0", 1.0)  # always solved
        src.registry.update("t1", 0.0)  # never solved
    assert src.weight("t0") == src.weight("t1") == 0.05
    assert src.weight("t2") == 1.0  # unseen
    src.registry.update("t2", 1.0)
    src.registry.update("t2", 0.0)
    assert src.weight("t2") == pytest.approx(1.0)  # p = 1/2 after one success, one failure


def test_batches_are_deterministic_distinct_and_fall_back_past_the_pool(tmp_path: Path) -> None:
    src = _source(tmp_path, 10, seed=3)
    a, b = src.batch(5, 4), src.batch(5, 4)
    assert [t.uid for t in a] == [t.uid for t in b]
    assert len({t.uid for t in a}) == 4
    assert len(src.batch(0, 25)) == 25  # base source's with-replacement draw


def test_frontier_tasks_are_drawn_far_more_often_than_saturated_ones(tmp_path: Path) -> None:
    src = _source(tmp_path, 20, seed=0, min_obs=4, floor=0.05)
    for i in range(20):
        for k in range(4):
            if i < 16:
                src.registry.update(f"t{i}", 1.0)  # 16 always-solved tasks
            else:
                src.registry.update(f"t{i}", float(k % 2))  # 4 tasks at p = 1/2
    drawn = Counter(t.uid for step in range(200) for t in src.batch(step, 4))
    frontier = sum(drawn[f"t{i}"] for i in range(16, 20))
    saturated = sum(drawn[f"t{i}"] for i in range(16))
    assert frontier > 3 * saturated  # 4 frontier tasks dominate 16 solved ones
    assert saturated > 0  # the floor still revisits solved tasks


def test_observe_feeds_measured_rewards_only(tmp_path: Path) -> None:
    src = _source(tmp_path, 2)
    rows = [SimpleNamespace(uid="t0", reward=1.0), SimpleNamespace(uid="t0", reward=None)]
    src.observe(rows)
    assert src.registry.counts == {"t0": 1}
    assert src.stats()["tasks_seen"] == 1


def test_bad_settings_are_refused(tmp_path: Path) -> None:
    with pytest.raises(TaskSourceRefusal, match="floor"):
        _source(tmp_path, 2, floor=0.0)
    with pytest.raises(TaskSourceRefusal, match="min_obs"):
        _source(tmp_path, 2, min_obs=0)


def test_a_rollout_updates_the_registry_and_logs_a_curriculum_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = _source(tmp_path, 4, seed=1)
    host = fakes._host(
        tmp_path,
        uids=["unused"],
        client_factory=lambda i: fakes.SubmitClient("xy"),
        tasks=src,
        gates=False,
    )
    host.rollout(0)
    assert sum(src.registry.counts.values()) == 2  # tasks_per_step=1 x group_size=2
    line = next(
        ln
        for ln in capsys.readouterr().err.splitlines()
        if ln.startswith("[agentic-rl:curriculum]")
    )
    assert json.loads(line.split(" ", 1)[1])["tasks_seen"] == 1


def test_config_keys_default_off_accept_frontier_and_refuse_bad_values() -> None:
    import importlib.util

    from foundationscale.agentic_rl.config import AgenticConfigRefusal, build_config

    spec = importlib.util.spec_from_file_location(
        "_test_config_raw_cur", Path(__file__).with_name("test_config.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def build(**rollout: Any) -> Any:
        raw = module._raw_config()
        raw["rollout"] = {**raw["rollout"], **rollout}
        return build_config(raw, {})

    assert build().rollout.curriculum == "off"
    cfg = build(curriculum="frontier", curriculum_min_obs=6, curriculum_floor=0.1).rollout
    assert (cfg.curriculum, cfg.curriculum_min_obs, cfg.curriculum_floor) == ("frontier", 6, 0.1)
    with pytest.raises(AgenticConfigRefusal, match="curriculum"):
        build(curriculum="easy-first")
    with pytest.raises(AgenticConfigRefusal, match="curriculum_floor"):
        build(curriculum_floor=0.0)


def test_registry_state_round_trips_and_resumes_a_source(tmp_path: Path) -> None:
    state = str(tmp_path / "curriculum.json")
    src = _source(tmp_path, 3, state_path=state)
    src.observe([SimpleNamespace(uid="t0", reward=1.0), SimpleNamespace(uid="t1", reward=0.0)])
    resumed = _source(tmp_path, 3, state_path=state)  # a restarted run on the same pool
    assert resumed.registry.counts == {"t0": 1, "t1": 1}
    assert resumed.registry.sums == {"t0": 1.0, "t1": 0.0}


def test_a_malformed_registry_file_is_refused_not_silently_emptied(tmp_path: Path) -> None:
    state = tmp_path / "curriculum.json"
    state.write_text(json.dumps({"counts": {"t0": 2}, "sums": {"t0": 5.0}}))  # sum > count
    with pytest.raises(TaskSourceRefusal, match="not a registry"):
        _source(tmp_path, 3, state_path=str(state))
    state.write_text(json.dumps([1, 2, 3]))  # not even an object
    with pytest.raises(TaskSourceRefusal, match="not a registry"):
        _source(tmp_path, 3, state_path=str(state))


def test_revisit_fraction_brings_frontier_tasks_back_even_in_a_huge_pool(tmp_path: Path) -> None:
    """GB200 math has 40k tasks: without a revisit share a measured frontier task competes with
    every unseen task and is practically never drawn again."""
    src = _source(tmp_path, 400, seed=0, revisit_fraction=0.5)
    for k in range(8):  # two frontier tasks (p ~ 1/2) and one saturated task
        src.registry.update("t7", float(k % 2))
        src.registry.update("t9", float(k % 2))
        src.registry.update("t11", 1.0)
    batch = [t.uid for t in src.batch(1, 4)]
    assert {"t7", "t9"} <= set(batch)  # both revisited (quota = 2)
    assert "t11" not in batch or src.weight("t11") == src.floor
    assert len(set(batch)) == 4
    none = _source(tmp_path, 400, seed=0, revisit_fraction=0.0)
    none.registry = src.registry
    drawn = Counter(t.uid for step in range(50) for t in none.batch(step, 4))
    assert drawn["t7"] + drawn["t9"] < 5  # without the share they are almost never drawn


def test_revisit_quota_falls_through_when_few_tasks_are_revisitable(tmp_path: Path) -> None:
    src = _source(tmp_path, 10, seed=0, revisit_fraction=1.0)
    for k in range(8):
        src.registry.update("t3", float(k % 2))  # one revisitable task, quota of 4
    batch = [t.uid for t in src.batch(0, 4)]
    assert batch[0] == "t3" and len(set(batch)) == 4
    with pytest.raises(TaskSourceRefusal, match="revisit_fraction"):
        _source(tmp_path, 10, revisit_fraction=1.5)


def test_review_fixes_empty_state_path_bool_sums_and_unmeasured_mean(tmp_path: Path) -> None:
    with pytest.raises(TaskSourceRefusal, match="state_path"):
        _source(tmp_path, 2, state_path="")
    bad = tmp_path / "bool.json"
    bad.write_text(json.dumps({"counts": {"u": 1}, "sums": {"u": True}}))
    with pytest.raises(TaskSourceRefusal, match="not a registry"):
        DifficultyRegistry.load(str(bad))
    assert _source(tmp_path, 2).stats()["mean_pass_rate"] is None  # nothing measured yet
