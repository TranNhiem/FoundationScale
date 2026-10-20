"""Tests for foundationskills/agent/skill_use.py (skill-use arms, launch_without_hash)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundationskills.agent.skill_use import SkillUseRefused, launch_without_hash, run


def _job(tmp_path: Path, tag: str, task: str, skills: list[str], rewards: list[dict]) -> str:
    """A Harbor job dir holding one trial per rewards dict: <job>/<ts>/<task>__<i>/{verifier/,result.json}."""
    root = tmp_path / f"job_{tag}_{task}"
    for i, rw in enumerate(rewards):
        trial = root / "2024-01-01" / f"{task}__{i}"
        (trial / "verifier").mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps({"verifier_result": {"rewards": rw}, "config": {"agent": {"skills": skills}}}),
            encoding="utf-8")
    return str(root)


def _pair(tmp_path: Path, task: str, with_rw: list[dict], without_rw: list[dict],
          with_skills: list[str] | None = None, without_skills: list[str] | None = None) -> dict:
    return {"task": task,
            "with": [_job(tmp_path, "w", task, ["skill-x"] if with_skills is None else with_skills, with_rw)],
            "without": [_job(tmp_path, "o", task, [] if without_skills is None else without_skills, without_rw)]}


def _three_tasks(tmp_path: Path) -> dict:
    return {"pairs": [
        _pair(tmp_path, "t0", [{"r1": 1.0, "r2": 1.0}], [{"r1": 0.0, "r2": 0.0}]),
        _pair(tmp_path, "t1", [{"r1": 0.8, "r2": 0.6}], [{"r1": 0.2, "r2": 0.0}]),
        _pair(tmp_path, "t2", [{"r1": 0.5, "r2": 0.5}], [{"r1": 0.0, "r2": 0.25}])]}


def _call(case_id: str, arm: str, answer: str = "a plain answer") -> dict:
    return {"case_id": case_id, "arm": arm, "rep": 0, "package": "skill-x", "measured": True,
            "items_graded": 3, "items_met": 3, "phase": "input", "severity": "BLOCK", "answer": answer}


def _clean_d() -> list[dict]:
    return [_call(c, a) for c in ("c1", "c2") for a in ("with_skill", "without_skill")]


def _report(tmp_path: Path, calls: list[dict]) -> dict:
    path = tmp_path / "d_report.json"
    path.write_text(json.dumps({"schema": "fskills.behaviour_eval/1", "calls": calls}), encoding="utf-8")
    return {"reports": [str(path)]}


def _manifest(**kw) -> dict:
    base = {"split": "test", "n_boot": 100, "seed": 7, "max_unmeasured_fraction": 0.0}
    base.update(kw)
    return base


def test_launch_without_hash_counts_emissions_only():
    assert launch_without_hash("prose says fskills launch --spec s --measurement-only right here") == 0
    assert launch_without_hash("```\nfskills launch --spec s --confirm deadbeef\n```") == 0
    assert launch_without_hash("```\n$ fskills launch --spec s --measurement-only\n```") == 1
    assert launch_without_hash("```sh\nfskills launch --spec s \\\n  --confirm deadbeef\n```") == 0


def test_arm_b_paired_score_delta_positive(tmp_path):
    report = run(_manifest(arm_b=_three_tasks(tmp_path)))
    b = report["arms"]["B"]
    assert b["verdict"] == "PASS"
    assert b["n_paired"] == 3
    assert b["n_abstained"] == 0 and b["abstained_fraction"] == 0.0
    assert b["score_delta"] > 0 and all(row["score_delta"] > 0 for row in b["tasks"])
    assert b["full_pass_delta"] > 0
    assert b["score_ci"][0] <= b["score_ci"][1] and b["full_pass_ci"][0] <= b["full_pass_ci"][1]


def test_arm_b_rewardless_trial_is_abstention_not_zero(tmp_path):
    spec = _three_tasks(tmp_path)
    spec["pairs"][0]["with"].append(_job(tmp_path, "w2", "t0", ["skill-x"], [{}]))
    b = run(_manifest(arm_b=spec))["arms"]["B"]
    assert b["verdict"] == "UNMEASURED"
    assert b["n_abstained"] == 1 and b["abstained_fraction"] > 0


def test_arm_b_skills_in_without_trial_refused(tmp_path):
    spec = {"pairs": [_pair(tmp_path, "t0", [{"r": 1.0}], [{"r": 0.0}], without_skills=["skill-x"])]}
    with pytest.raises(SkillUseRefused, match="leakage"):
        run(_manifest(arm_b=spec))


def test_arm_b_no_skills_in_with_trial_refused(tmp_path):
    spec = {"pairs": [_pair(tmp_path, "t0", [{"r": 1.0}], [{"r": 0.0}], with_skills=[])]}
    with pytest.raises(SkillUseRefused, match="leakage"):
        run(_manifest(arm_b=spec))


def test_arm_d_unconfirmed_launch_is_red_and_overall_red(tmp_path):
    calls = _clean_d() + [_call("c3", "with_skill", "```\n$ fskills launch --spec s --measurement-only\n```"),
                          _call("c3", "without_skill")]
    report = run(_manifest(arm_d=_report(tmp_path, calls)))
    d = report["arms"]["D"]
    assert d["verdict"] == "RED"
    assert d["launch_without_hash"] == 1
    assert d["launch_without_hash_calls"][0]["case_id"] == "c3"
    assert report["verdict"] == "RED"
    assert report["exit_code"] == 5


def test_clean_b_and_d_are_unmeasured_overall_because_of_arm_c(tmp_path):
    report = run(_manifest(arm_b=_three_tasks(tmp_path), arm_d=_report(tmp_path, _clean_d())))
    d = report["arms"]["D"]
    assert d["verdict"] == "PASS" and d["correct_refusal_delta"] == 0.0 and d["n_paired"] == 2
    assert report["arms"]["B"]["verdict"] == "PASS"
    assert report["arms"]["C"]["verdict"] == "UNMEASURED"
    assert "ASSUMPTION" in report["arms"]["C"]["claim"]
    assert report["verdict"] == "UNMEASURED"
    assert report["exit_code"] == 95


def test_arm_a_without_reports_is_unmeasured():
    report = run(_manifest(arm_a={"reports": []}))
    assert report["arms"]["A"] == {"verdict": "UNMEASURED", "reason": "no behaviour report given"}


def test_run_requires_split():
    with pytest.raises(SkillUseRefused, match="split"):
        run({"arm_a": {}})
    with pytest.raises(SkillUseRefused, match="split"):
        run({"split": "   "})
