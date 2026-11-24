"""CurriculumTaskSource (atlas v2 adoption B): informative-first draws, quarantine of flat tasks, infra ignored."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from foundationscale.agentic_rl.tasks import CurriculumTaskSource, TaskSource, TaskSourceRefusal


@dataclass
class T:
    uid: str
    reward: float | None


def _pool(tmp_path, n=20):
    p = tmp_path / "tasks.jsonl"
    p.write_text("".join(json.dumps({"uid": f"u{i}", "messages": [{"role": "user", "content": "x"}]}) + "\n"
                         for i in range(n)), encoding="utf-8")
    return str(p)


def _group(uid, rewards):
    return [T(uid, r) for r in rewards]


def test_unobserved_draw_matches_the_uniform_source(tmp_path):
    path = _pool(tmp_path)
    assert CurriculumTaskSource(path=path, seed=3).batch(0, 4) == TaskSource(path=path, seed=3).batch(0, 4)


def test_informative_tasks_fill_the_informative_share(tmp_path):
    src = CurriculumTaskSource(path=_pool(tmp_path), seed=1)
    src.observe(0, _group("u5", [0.0, 0.4]) + _group("u6", [0.2, 0.2]) + _group("u7", [0.0, 0.0]))
    for step in range(1, 30):
        uids = [t.uid for t in src.batch(step, 4)]
        assert sum(u in ("u5", "u6") for u in uids) >= 2  # round(0.8 * 4) = 3, capped by the 2 informative tasks
        assert len(set(uids)) == 4


def test_flat_tasks_are_quarantined_then_reprobed(tmp_path):
    src = CurriculumTaskSource(path=_pool(tmp_path, 6), seed=0, quarantine_after=2, reprobe_every=5)
    src.observe(0, _group("u0", [0.0, 0.0]))
    summary = src.observe(1, _group("u0", [0.0, 0.0]))
    assert summary["newly_quarantined"] == 1
    assert all("u0" not in [t.uid for t in src.batch(s, 5)] for s in range(2, 7))
    assert any("u0" in [t.uid for t in src.batch(s, 5)] for s in range(7, 30))


def test_infra_abstentions_change_nothing(tmp_path):
    src = CurriculumTaskSource(path=_pool(tmp_path), seed=0)
    assert src.observe(0, _group("u1", [None, None]))["observed"] == 0 and src.batch(1, 4) == TaskSource(
        path=src.path, seed=0).batch(1, 4)


def test_bad_parameters_are_refused(tmp_path):
    with pytest.raises(TaskSourceRefusal):
        CurriculumTaskSource(path=_pool(tmp_path), informative_share=1.5)


def test_config_flag_selects_the_curriculum_source(tmp_path):
    from foundationscale.agentic_rl import cli
    from foundationscale.agentic_rl.config import RolloutConfig

    class C:
        rollout = RolloutConfig(tasks_path=_pool(tmp_path), tasks_per_step=4, group_size=8, max_concurrency=4, curriculum=True)

    assert isinstance(cli._task_source(C), CurriculumTaskSource)
    C.rollout = RolloutConfig(tasks_path=_pool(tmp_path), tasks_per_step=4, group_size=8, max_concurrency=4)
    assert type(cli._task_source(C)) is TaskSource
