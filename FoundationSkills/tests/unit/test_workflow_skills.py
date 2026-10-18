"""A3.2 workflow skills /fs-data, /fs-eval, /fs-plan: input refusals fire from real requests; a composed step's outcome
maps onto the caller's own rules (REFUSED / RED / UNMEASURED), with the inner findings as evidence, never as findings."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from foundationskills.core.artifacts import ArtifactRef
from foundationskills.core.contract import SkillContext
from foundationskills.core.status import Status
from foundationskills.skills import _workflow
from foundationskills.skills.fs_data.skill import FsDataSkill
from foundationskills.skills.fs_eval.skill import FsEvalSkill
from foundationskills.skills.fs_plan.skill import FsPlanSkill


def _fake_step(status: Status, artifacts: tuple = ()):
    def fake(caller, name, request, ctx, reg=None):
        assert name in caller.invokes
        result = SimpleNamespace(status=status, refusal="inner refusal" if status is Status.REFUSED else None,
                                 artifacts=artifacts, findings=())
        step = _workflow.Step(skill=name, status=status.value, exit_code=status.exit_code, refusal=result.refusal,
                              findings=[{"rule_id": "XX-IN-001", "severity": "BLOCK", "message": "m", "recovery": ""}])
        return result, step
    return fake


@pytest.mark.parametrize(("skill", "rule"), [(FsDataSkill(), "FD-IN-001"), (FsEvalSkill(), "FE-IN-001"),
                                             (FsPlanSkill(), "FP-IN-001")])
def test_an_empty_request_is_refused_by_the_input_rule(skill, rule, tmp_path) -> None:
    result = skill.execute({}, SkillContext(workdir=tmp_path))
    assert result.status is Status.REFUSED and rule in [f.rule_id for f in result.findings]


def test_fs_plan_without_a_probe_report_is_refused(tmp_path) -> None:
    result = FsPlanSkill().execute({"goal": "goal.json"}, SkillContext(workdir=tmp_path))
    assert result.status is Status.REFUSED and "FP-IN-002" in [f.rule_id for f in result.findings]


CASES = [
    (FsDataSkill, {"sources": [{"uri": "/d", "kind": "jsonl"}]}, "fs_data", "FD"),
    (FsEvalSkill, {"checkpoint": "/models/m"}, "fs_eval", "FE"),
]


@pytest.mark.parametrize(("cls", "request_", "module", "prefix"), CASES)
@pytest.mark.parametrize(("inner", "status", "rule"), [(Status.REFUSED, Status.REFUSED, "IN-002"),
                                                       (Status.RED, Status.RED, "HO-001"),
                                                       (Status.UNMEASURED, Status.UNMEASURED, "HO-002")])
def test_a_composed_outcome_maps_onto_the_callers_rule(cls, request_, module, prefix, inner, status, rule,
                                                       tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(f"foundationskills.skills.{module}.skill.run_step", _fake_step(inner))
    result = cls().execute(dict(request_), SkillContext(workdir=tmp_path))
    ids = [f.rule_id for f in result.findings]
    assert result.status is status and f"{prefix}-{rule}" in ids, (result.status, ids)
    assert "XX-IN-001" not in ids  # the inner rule id travels as evidence only


def test_fs_plan_shows_the_plan_hash_and_reds_a_non_executable_spec(tmp_path, monkeypatch) -> None:
    for name, payload in (("goal.json", {"base_model": {}}), ("probe.json", {}), ("plan.json", {"stages": []})):
        (tmp_path / name).write_text(json.dumps(payload))
    spec_ok = tmp_path / "spec_ok.json"; spec_ok.write_text(json.dumps({"executable": True, "stage": "sft"}))
    spec_bad = tmp_path / "spec_bad.json"; spec_bad.write_text(json.dumps({"executable": False, "missing": "x"}))
    artifacts = {"training.planner": (ArtifactRef("training_plan", "p", str(tmp_path / "plan.json"), "0"),),
                 "probe_first": (), "training.emit": None}

    def run_with(specs):
        artifacts["training.emit"] = tuple(ArtifactRef("fs_launch_spec", s.stem, str(s), "0") for s in specs)

        def fake(caller, name, request, ctx, reg=None):
            return _fake_step(Status.PASS, artifacts[name])(caller, name, request, ctx, reg)
        monkeypatch.setattr("foundationskills.skills.fs_plan.skill.run_step", fake)
        return FsPlanSkill().execute({"goal": str(tmp_path / "goal.json"), "probe_report": str(tmp_path / "probe.json"),
                                      "model": "/m"}, SkillContext(workdir=tmp_path))

    good = run_with([spec_ok])
    assert good.status is Status.PASS and good.payload["outputs"]["launch_specs"][0]["plan_hash"]
    assert "type it back to /fs-launch" in good.payload["show_user"]
    bad = run_with([spec_bad])
    assert bad.status is Status.RED and "FP-HO-001" in [f.rule_id for f in bad.findings]


def test_fs_eval_writes_its_record_beside_not_into_the_evaluation_out_file(tmp_path, monkeypatch) -> None:
    # measured 2026-10-18 on GPU: `out` (evaluation's report FILE) was reused as the record dir; the evaluation
    # passed (arc_easy 0.473 -> 0.533) and /fs-eval then crashed writing its workflow_record
    report = tmp_path / "eval" / "arc_easy.json"
    report.parent.mkdir()
    report.write_text("{}")
    monkeypatch.setattr("foundationskills.skills.fs_eval.skill.run_step",
                        _fake_step(Status.PASS, (ArtifactRef("eval_report", "e", str(report), "0"),)))
    result = FsEvalSkill().execute({"run_manifest": "/x/run_manifest.json", "out": str(report)},
                                   SkillContext(workdir=tmp_path))
    assert result.status is Status.PASS, [f.message for f in result.findings]
    assert all(Path(ref.path).parent != report for ref in result.artifacts)
