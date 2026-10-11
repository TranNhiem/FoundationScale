"""Exit-code matrix and must-fire fixtures for ProbeFirstSkill (tmp_path only, no FoundationScale execution)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foundationskills.core.contract import SkillContext, SkillResult
from foundationskills.core.status import Status
from foundationskills.skills.probe_first import skill as pf_skill
from foundationskills.skills.probe_first.skill import ProbeFirstSkill, caps_from_probe, probe_payload

EXPECTATIONS = {
    "PF-IN-001": (Status.REFUSED, 96),
    "PF-IN-002": (Status.REFUSED, 96),
    "PF-IN-003": (Status.REFUSED, 96),
    "PF-HO-001": (Status.UNMEASURED, 95),
    "PF-HO-002": (Status.RED, 5),
}


def materialize(
    request: dict[str, Any],
    probe: dict[str, Any] | None,
    plan: dict[str, Any],
    tmp_path: Path,
    envelope: bool = False,
) -> tuple[dict[str, Any], SkillContext]:
    """Create plan.json / probe.json under tmp_path and return real request paths + a tmp context."""
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    real: dict[str, Any] = dict(request)
    real["plan_draft"] = str(plan_path)
    real["out"] = str(tmp_path / str(request.get("out", "artifacts")))
    if probe is None:
        real.pop("probe_report", None)
    else:
        raw: Any = {"type": "probe_report", "id": "probe", "payload": probe} if envelope else probe
        probe_path = tmp_path / "probe.json"
        probe_path.write_text(json.dumps(raw), encoding="utf-8")
        real["probe_report"] = str(probe_path)
    return real, SkillContext(tmp_path)


def one_check() -> dict[str, Any]:
    return {"stage": "sft", "backend": "fsdp"}


# -- declared contract ------------------------------------------------------------------------
def test_five_rules_with_the_contract_severities_and_phases() -> None:
    skill = ProbeFirstSkill()
    assert skill.name == "probe_first"
    by_id = {spec.rule_id: spec for spec in skill.rules}
    assert set(by_id) == {"PF-IN-001", "PF-IN-002", "PF-IN-003", "PF-HO-001", "PF-HO-002"}
    for index in ("001", "002", "003"):
        assert by_id[f"PF-IN-{index}"].severity.value == "BLOCK"
        assert by_id[f"PF-IN-{index}"].phase == "input"
    assert by_id["PF-HO-001"].severity.value == "WARN"
    assert by_id["PF-HO-001"].phase == "handoff"
    assert by_id["PF-HO-002"].severity.value == "BLOCK"
    assert by_id["PF-HO-002"].phase == "handoff"


def test_must_fire_fixtures_cover_every_declared_rule() -> None:
    skill = ProbeFirstSkill()
    fixtures = skill.must_fire_fixtures()
    assert set(fixtures) == {spec.rule_id for spec in skill.rules}
    for rule_id, fixture in fixtures.items():
        assert rule_id in fixture["expected"], fixture


def test_probe_payload_round_trips_through_caps_from_probe() -> None:
    caps = caps_from_probe(probe_payload(train_objectives=("sft",), errors=("probe: deep axis measurement failed",)))
    assert caps.available is True
    assert caps.backends == ("fsdp",)
    assert caps.errors == ("probe: deep axis measurement failed",)
    assert caps.check(stage="sft", backend="fsdp") is None


# -- must-fire matrix --------------------------------------------------------------------------
@pytest.mark.parametrize("rule_id", ["PF-IN-001", "PF-IN-002", "PF-IN-003", "PF-HO-001", "PF-HO-002"])
def test_must_fire_matrix(rule_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    skill = ProbeFirstSkill()
    fixture = skill.must_fire_fixtures()[rule_id]
    request, ctx = materialize(fixture["request"], fixture.get("probe"), fixture.get("plan", {}), tmp_path)
    if fixture.get("tamper"):
        real_write = pf_skill.write_artifact

        def write_and_lose(artifact, directory):  # "delete report before handoff"
            ref = real_write(artifact, directory)
            Path(ref.path).unlink()
            return ref

        monkeypatch.setattr(pf_skill, "write_artifact", write_and_lose)
    result = skill.execute(request, ctx)
    status, code = EXPECTATIONS[rule_id]
    assert result.status is status, fixture["expected"]
    assert result.exit_code == code
    assert rule_id in {finding.rule_id for finding in result.findings}
    out_dir = Path(request["out"])
    on_disk = sorted(out_dir.glob("probe_report.*.json")) if out_dir.exists() else []
    if code == 96:  # refusals write nothing
        assert result.refusal and rule_id in result.refusal
        assert on_disk == []
        assert result.artifacts == ()
    else:
        assert result.refusal is None
        assert result.artifacts and result.artifacts[0].type == "probe_report"
        expected_files = [] if fixture.get("tamper") else [Path(result.artifacts[0].path).name]
        assert [path.name for path in on_disk] == expected_files


# -- rule by rule --------------------------------------------------------------------------------
def test_pf_in_001_probe_absent_unreadable_or_not_a_probe(tmp_path: Path) -> None:
    request, ctx = materialize({"plan_draft": "plan.json", "out": "artifacts"}, None, {"checks": [one_check()]}, tmp_path)
    request["probe_report"] = str(tmp_path / "does-not-exist.json")  # unreadable
    result = ProbeFirstSkill().execute(request, ctx)
    assert (result.status, result.exit_code) == (Status.REFUSED, 96)
    assert "PF-IN-001: missing input" in result.refusal

    request, ctx = materialize({"plan_draft": "plan.json", "out": "artifacts"}, {"not": "a probe"}, {"checks": [one_check()]}, tmp_path)
    result = ProbeFirstSkill().execute(request, ctx)
    assert result.status is Status.REFUSED
    assert "PF-IN-001" in result.refusal and "the probe was skipped" in result.refusal


def test_pf_in_002_treat_unmeasured_as_pass_carries_the_missing_string(tmp_path: Path) -> None:
    request, ctx = materialize(
        {"plan_draft": "plan.json", "checks": [{"stage": "sft", "backend": "fsdp", "pp": 2}], "out": "artifacts"},
        # axes_measured=False: pp is unmeasured; train_objectives=("sft",) keeps the
        # documented sft gap closed so the unmeasured axis is the ONLY gap `check` can hit
        probe_payload(train_objectives=("sft",)),
        {"checks": [{"stage": "sft", "backend": "fsdp", "pp": 2, "status": "PASS"}]},
        tmp_path,
    )
    result = ProbeFirstSkill().execute(request, ctx)
    assert (result.status, result.exit_code) == (Status.REFUSED, 96)
    findings = [finding for finding in result.findings if finding.rule_id == "PF-IN-002"]
    assert findings, result.to_dict()
    assert findings[0].evidence["missing"].startswith("missing: pp")
    assert findings[0].evidence["plan_claims_pass"] is True
    assert "treat UNMEASURED as PASS" in result.refusal
    out_dir = Path(request["out"])
    assert not out_dir.exists() or not list(out_dir.glob("probe_report.*.json"))


def test_pf_in_003_no_requested_checks_named(tmp_path: Path) -> None:
    request, ctx = materialize({"plan_draft": "plan.json", "out": "artifacts"}, probe_payload(), {}, tmp_path)
    result = ProbeFirstSkill().execute(request, ctx)
    assert (result.status, result.exit_code) == (Status.REFUSED, 96)
    assert "PF-IN-003" in result.refusal


def test_pf_ho_001_partial_probe_is_unmeasured_and_never_pass(tmp_path: Path) -> None:
    request, ctx = materialize(
        {"plan_draft": "plan.json", "checks": [one_check()], "out": "artifacts"},
        probe_payload(train_objectives=("sft",), errors=("probe: deep axis measurement failed",)),
        {"checks": [one_check()]},
        tmp_path,
    )
    result = ProbeFirstSkill().execute(request, ctx)
    assert (result.status, result.exit_code) == (Status.UNMEASURED, 95)
    assert {finding.rule_id for finding in result.findings} == {"PF-HO-001"}
    assert result.findings[0].severity.value == "WARN"
    assert result.payload["verdict"] == "UNMEASURED"
    assert result.payload["checks"] == [{"check": one_check(), "missing": None}]
    assert (tmp_path / "artifacts" / "probe_report.plan.json").is_file()


def test_pf_ho_002_handoff_catches_an_altered_report(tmp_path: Path) -> None:
    request, ctx = materialize(
        {"plan_draft": "plan.json", "checks": [one_check()], "out": "artifacts"},
        probe_payload(train_objectives=("sft",)),
        {"checks": [one_check()]},
        tmp_path,
    )
    skill = ProbeFirstSkill()
    result = skill.execute(request, ctx)
    assert result.status is Status.PASS
    report = tmp_path / "artifacts" / "probe_report.plan.json"
    report.write_text(report.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    findings = skill.check_handoff(result, ctx)
    assert [finding.rule_id for finding in findings] == ["PF-HO-002"]
    assert "changed after it was written" in findings[0].message


def test_pass_writes_vetted_probe_report_with_provenance(tmp_path: Path) -> None:
    request, ctx = materialize(
        {"plan_draft": "plan.json", "checks": [one_check()], "out": "artifacts", "artifact_id": "q27"},
        probe_payload(train_objectives=("sft",)),
        {"checks": [one_check()]},
        tmp_path,
    )
    result = ProbeFirstSkill().execute(request, ctx)
    assert (result.status, result.exit_code) == (Status.PASS, 0)
    assert result.refusal is None and result.findings == ()
    assert result.payload["verdict"] == "PASS"
    assert result.payload["checks"] == [{"check": one_check(), "missing": None}]
    assert result.payload["capabilities"]["available"] is True
    assert result.payload["plan_draft"] == request["plan_draft"]
    written_path = tmp_path / "artifacts" / "probe_report.q27.json"
    assert written_path.is_file()
    written = json.loads(written_path.read_text(encoding="utf-8"))
    assert written["provenance"]["skill"] == "probe_first"
    assert written["provenance"]["skill_version"] == "0.1.0"
    assert written["payload"]["verdict"] == "PASS"
    assert result.provenance is not None and result.provenance.skill == "probe_first"
    assert result.artifacts[0].sha256 == written["provenance"] or result.artifacts[0].type == "probe_report"


def test_probe_json_accepts_an_artifact_envelope(tmp_path: Path) -> None:
    request, ctx = materialize(
        {"plan_draft": "plan.json", "checks": [one_check()], "out": "artifacts"},
        probe_payload(train_objectives=("sft",)),
        {"checks": [one_check()]},
        tmp_path,
        envelope=True,
    )
    result = ProbeFirstSkill().execute(request, ctx)
    assert (result.status, result.exit_code) == (Status.PASS, 0)
    assert result.payload["capabilities"]["available"] is True


def test_missing_plan_draft_is_refused_as_invalid_request(tmp_path: Path) -> None:
    result = ProbeFirstSkill().execute({"out": str(tmp_path / "artifacts")}, SkillContext(tmp_path))
    assert (result.status, result.exit_code) == (Status.REFUSED, 96)
    assert result.refusal.startswith("invalid request")


def test_diagnose_names_the_probe_step_for_every_status() -> None:
    skill = ProbeFirstSkill()
    exc_diag = skill.diagnose(RuntimeError("boom"))
    assert exc_diag.symptom.startswith("probe-first RuntimeError")
    assert any("fskills probe --deep --json" in cause for cause in exc_diag.likely_causes)
    assert any("fskills probe --deep --json" in step for step in exc_diag.recovery)

    result_diag = skill.diagnose(SkillResult(status=Status.UNMEASURED, payload={}))
    assert result_diag.symptom.startswith("probe-first UNMEASURED")
    assert result_diag.checks and result_diag.recovery


def test_a_planner_training_plan_without_checks_is_vetted_per_stage(tmp_path: Path) -> None:
    """`fskills plan` emits `stages`, not `checks` (found on the A3.1 GPU chain): each stage becomes a check."""
    plan = {"stages": [{"stage": "sft", "algorithm": "sft", "method": "lora", "multi_gpu_rl": True}]}
    request, ctx = materialize({"plan_draft": "plan.json", "out": "artifacts"},
                               probe_payload(train_objectives=("sft",)), plan, tmp_path)
    result = ProbeFirstSkill().execute(request, ctx)
    assert result.status == Status.PASS, [f.rule_id for f in result.findings]
    # only stage/algorithm are lifted from a stage: a planner stage never asserts axes or multi-GPU RL
    assert [row["check"] for row in result.payload["checks"]] == [{"stage": "sft", "algorithm": "sft"}]
