"""A4.5 harbor suite: measured-only pass@1, the paper denominator, nop negative controls, the skill verdicts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foundationskills.core.contract import SkillContext
from foundationskills.core.status import Status
from foundationskills.skills.evaluation.skill import EvalSkill
from foundationskills.skills.evaluation.suites.harbor import HarborSuiteRefused, score


def _trial(job: Path, task: str, n: int, rewards: dict[str, Any] | None, exc: str | None = None,
           agent: str = "pi") -> None:
    trial = job / f"{task}__t{n}"
    (trial / "verifier").mkdir(parents=True)
    record = {"task_name": f"terminal-bench/{task}", "verifier_result": {"rewards": rewards} if rewards else None,
              "agent_info": {"name": "nop" if job.name.startswith("nop") else agent},
              "exception_info": {"exception_type": exc} if exc else None}
    (trial / "result.json").write_text(json.dumps(record), encoding="utf-8")


def _jobs(tmp_path: Path) -> tuple[Path, Path]:
    job, nop = tmp_path / "job", tmp_path / "nop"
    _trial(job, "a", 0, {"reward": 1.0})
    _trial(job, "a", 1, {"reward": 0.0}, exc="AgentTimeoutError")  # timed out but measured: a fail
    _trial(job, "b", 0, {"reward": 1.0, "style": 1.0})
    _trial(job, "b", 1, None, exc="SandboxError")  # infra: unmeasured, never a 0
    _trial(job, "c", 0, None, exc="SandboxError")  # a task with no measured trial
    (job / "result.json").write_text("{}", encoding="utf-8")  # job-level summary: not a trial
    for task in ("a", "b", "c"):
        _trial(nop, task, 0, {"reward": 0.0})
    return job, nop


def test_measured_only_pass_at_1_and_the_paper_denominator(tmp_path: Path) -> None:
    job, nop = _jobs(tmp_path)
    report = score([str(job)], [str(nop)], tasks=["a", "b", "c", "d"], max_unmeasured_fraction=1.0)

    assert report["pass_at_1"] == 75.0  # a: 1/2, b: 1/1 (the infra trial is excluded); c and d have no measured trial
    assert report["tasks_scored"] == 2 and report["tasks_total"] == 4
    assert report["paper_denominator"]["pass_at_1"] == 25.0  # (1/2 + 1/2 + 0 + 0) / 4: B2.2's registered rule
    assert (report["unmeasured_cells"], report["cells"]) == (3, 6)  # b's and c's infra trials, plus trial-less d
    rows = {r["task"]: r for r in report["tasks"]}
    assert rows["c"]["dropped"] == "no_measured_trial" and rows["d"]["trials"] == 0
    assert report["verdict"] == "PASS"


def test_unmeasured_cells_over_the_declared_maximum_are_unmeasured(tmp_path: Path) -> None:
    job, nop = _jobs(tmp_path)
    assert score([str(job)], [str(nop)])["verdict"] == "UNMEASURED"  # default max 0: 2 of 5 cells unmeasured


def test_a_passing_negative_control_drops_its_task(tmp_path: Path) -> None:
    job, nop = _jobs(tmp_path)
    _trial(nop, "b", 1, {"reward": 1.0})  # the nop agent passed b: b's verifier cannot discriminate
    report = score([str(job)], [str(nop)], max_unmeasured_fraction=1.0)

    assert report["negative_controls"]["non_discriminating"] == ["b"]
    assert report["pass_at_1"] == 50.0 and report["tasks_scored"] == 1


def test_controls_from_a_solving_agent_refuse(tmp_path: Path) -> None:
    job, _ = _jobs(tmp_path)
    oracle = tmp_path / "oracle"
    _trial(oracle, "a", 0, {"reward": 1.0}, agent="oracle")
    _trial(oracle, "b", 0, {"reward": 0.0}, agent="oracle")
    with pytest.raises(HarborSuiteRefused, match="nop agent"):
        score([str(job)], [str(oracle)])


@pytest.mark.parametrize("controls", ["none", "unmeasured", "all_pass"])
def test_unusable_negative_controls_refuse(tmp_path: Path, controls: str) -> None:
    job, _ = _jobs(tmp_path)
    nop = tmp_path / "nop2"
    if controls == "unmeasured":
        _trial(nop, "a", 0, None, exc="VerifierTimeoutError")
    elif controls == "all_pass":
        _trial(nop, "a", 0, {"reward": 1.0})
    with pytest.raises(HarborSuiteRefused):
        score([str(job)], [str(nop)] if controls != "none" else [])


def test_a_job_without_trials_refuses(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    _, nop = _jobs(tmp_path)
    with pytest.raises(HarborSuiteRefused):
        score([str(tmp_path / "empty")], [str(nop)])


def _run(tmp_path: Path, **over: Any):
    job, nop = _jobs(tmp_path)
    request = {"suite": "harbor", "job_dirs": [str(job)], "control_dirs": [str(nop)],
               "out": str(tmp_path / "eval" / "harbor.json"), **over}
    return EvalSkill().execute(request, SkillContext(workdir=tmp_path))


def test_the_skill_reports_unmeasured_with_ev_ho_010(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.status is Status.UNMEASURED and result.exit_code == 95
    assert "EV-HO-010" in {f.rule_id for f in result.findings}
    assert Path(result.artifacts[0].path).is_file()


def test_the_skill_passes_within_the_declared_tolerance(tmp_path: Path) -> None:
    result = _run(tmp_path, max_unmeasured_fraction=0.5)
    assert result.status is Status.PASS, [(f.rule_id, f.message) for f in result.findings]
    row = result.payload["benchmarks"][0]
    assert (row["name"], row["score"], row["baseline"]) == ("harbor", 75.0, None)


@pytest.mark.parametrize("over, rule", [({"control_dirs": []}, "EV-IN-011"), ({"job_dirs": ["/nonexistent/job"]}, "EV-IN-010")])
def test_the_skill_refuses_unusable_inputs(tmp_path: Path, over: dict[str, Any], rule: str) -> None:
    result = _run(tmp_path, **over)
    assert result.status is Status.REFUSED and result.exit_code == 96
    assert rule in (result.refusal or "")


# == skill-behaviour suite ======================================================================================

from foundationskills.skills.evaluation.suites.skill_behaviour import SkillBehaviourRefused  # noqa: E402
from foundationskills.skills.evaluation.suites.skill_behaviour import score as behaviour_score  # noqa: E402


def _call(case: str, met: int, measured: bool = True, phase: str = "input", severity: str = "BLOCK",
          arm: str = "with_skill") -> dict[str, Any]:
    return {"case_id": case, "arm": arm, "package": "fskills-x", "measured": measured, "items_met": met,
            "items_graded": 3, "phase": phase, "severity": severity}


def _report(tmp_path: Path, calls: list[dict[str, Any]], rate: float | None, name: str = "r.json") -> str:
    results = {"with_skill": {"packages": {"fskills-x": {"item_pass_rate": rate}}}} if rate is not None else {}
    path = tmp_path / name
    path.write_text(json.dumps({"gated_arm": "with_skill", "cases": "oracle", "calls": calls, "results": results}),
                    encoding="utf-8")
    return str(path)


def test_behaviour_suite_reproduces_the_report_and_counts_refused_fixtures(tmp_path: Path) -> None:
    calls = [_call("a", 3), _call("b", 1, phase="handoff", severity="WARN"), _call("c", 3, arm="without_skill")]
    report = behaviour_score([_report(tmp_path, calls, 4 / 6)])

    assert report["item_pass_rate"] == round(4 / 6, 4) and report["refused_fixture_calls"] == 1
    pkg = report["packages"]["fskills-x:oracle"]
    assert (pkg["measured"], pkg["passes"], pkg["case_pass_rate"]) == (2, 1, 0.5)
    assert report["verdict"] == "PASS"


def test_behaviour_suite_unmeasured_calls_are_never_zeros(tmp_path: Path) -> None:
    calls = [_call("a", 3), _call("b", 0, measured=False)]
    report = behaviour_score([_report(tmp_path, calls, 1.0)])
    assert report["item_pass_rate"] == 1.0 and report["verdict"] == "UNMEASURED"
    assert behaviour_score([_report(tmp_path, calls, 1.0)], max_unmeasured_fraction=0.5)["verdict"] == "PASS"


@pytest.mark.parametrize("calls, rate", [
    ([_call("a", 3, phase="handoff", severity="WARN")], 1.0),  # no refused-fixture item
    ([_call("a", 3)], 0.5),  # the re-score does not reproduce the report
    ([_call("a", 3, arm="without_skill")], None),  # no gated-arm call
])
def test_behaviour_suite_refuses_unusable_reports(tmp_path: Path, calls: list[dict[str, Any]], rate: float | None) -> None:
    with pytest.raises(SkillBehaviourRefused):
        behaviour_score([_report(tmp_path, calls, rate)])


def test_the_skill_scores_behaviour_reports(tmp_path: Path) -> None:
    path = _report(tmp_path, [_call("a", 3)], 1.0)
    result = EvalSkill().execute({"suite": "skill-behaviour", "reports": [path], "out": str(tmp_path / "o.json")},
                                 SkillContext(workdir=tmp_path))
    assert result.status is Status.PASS, [(f.rule_id, f.message) for f in result.findings]
    assert result.payload["benchmarks"][0]["name"] == "skill-behaviour/fskills-x:oracle"
    bad = _report(tmp_path, [_call("a", 3, phase="handoff", severity="WARN")], 1.0, name="routing_only.json")
    refused = EvalSkill().execute({"suite": "skill-behaviour", "reports": [bad], "out": str(tmp_path / "p.json")},
                                  SkillContext(workdir=tmp_path))
    assert refused.status is Status.REFUSED and "EV-IN-011" in (refused.refusal or "")
