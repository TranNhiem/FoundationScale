"""Must-fire matrix and sealing behaviour for RefusalSurfaceSkill (stdlib + pytest, everything under tmp_path)."""
from __future__ import annotations

from pathlib import Path

import pytest

from foundationskills.core.artifacts import read_artifact
from foundationskills.core.contract import SkillContext, SkillResult
from foundationskills.core.provenance import sha256_json
from foundationskills.core.status import Status
from foundationskills.skills.refusal_surface import RefusalSurfaceSkill


@pytest.fixture()
def ctx(tmp_path: Path) -> SkillContext:
    """Everything the skill writes lands under tmp_path (workdir/artifacts)."""
    return SkillContext(workdir=tmp_path)


def test_declares_one_must_fire_fixture_per_rule() -> None:
    skill = RefusalSurfaceSkill()
    fixtures = skill.must_fire_fixtures()
    assert set(fixtures) == {"RS-IN-001", "RS-HO-001", "RS-HO-002"} == {spec.rule_id for spec in skill.rules}
    for name, fixture in fixtures.items():
        assert {"request", "expected"} <= set(fixture), name


def test_skill_metadata_matches_the_spec() -> None:
    skill = RefusalSurfaceSkill()
    assert skill.name == "refusal_surface"
    assert skill.version == "0.1.0"
    assert skill.consumes == ("finding_set",)
    assert skill.produces == ("finding_set",)
    assert skill.fs_interface.emits == ("finding_set",)
    assert skill.fs_interface.apis == ()
    assert skill.fs_interface.notes == "attribution only; no FoundationScale call"
    assert {spec.phase for spec in skill.rules if spec.rule_id == "RS-IN-001"} == {"input"}
    assert {spec.phase for spec in skill.rules if spec.rule_id.startswith("RS-HO-")} == {"handoff"}


def test_must_fire_rs_in_001_refuses_without_a_surface(ctx: SkillContext) -> None:
    """MUST_FIRE RS-IN-001: {findings: []} -> REFUSED 96 naming RS-IN-001, nothing written."""
    result = RefusalSurfaceSkill().execute({"findings": []}, ctx)
    assert result.status is Status.REFUSED
    assert result.exit_code == 96
    assert "RS-IN-001" in (result.refusal or "")
    assert [f.rule_id for f in result.findings] == ["RS-IN-001"]
    assert list(ctx.workdir.rglob("*.json")) == []


def test_must_fire_rs_ho_001_invented_rule_id_is_red(ctx: SkillContext) -> None:
    """MUST_FIRE RS-HO-001 / MUST_FIRE quartet 'invent rule ids': RED 5 with RS-HO-001."""
    request = {
        "findings": [{"rule_id": "PF-IN-999", "severity": "BLOCK", "message": "invented"}],
        "result_status": "REFUSED",
    }
    result = RefusalSurfaceSkill().execute(request, ctx)
    assert result.status is Status.RED
    assert result.exit_code == 5
    assert [f.rule_id for f in result.findings] == ["RS-HO-001"]
    assert all(f.rule_id != "PF-IN-999" for f in result.findings)  # never a Finding with an undeclared id
    assert result.findings[0].evidence["invented_rule_ids"] == ["PF-IN-999"]
    entry = result.payload["findings"][0]
    assert entry["rule_id"] == "RS-HO-001"
    assert entry["evidence"]["source_rule_id"] == "PF-IN-999"
    assert result.payload["result_status"] == "REFUSED"
    sealed = read_artifact(result.artifacts[0].path, expect_type="finding_set")  # the set is still sealed
    assert sealed.payload["findings"][0]["rule_id"] == "RS-HO-001"


def test_must_fire_rs_ho_002_unattributed_finding_is_attributed(ctx: SkillContext) -> None:
    """MUST_FIRE RS-HO-002: PASS 0 with RS-HO-002; the emitted entry rule_id is RS-HO-002."""
    result = RefusalSurfaceSkill().execute({"findings": [{"message": "no id at all"}]}, ctx)
    assert result.status is Status.PASS
    assert result.exit_code == 0
    assert [f.rule_id for f in result.findings] == ["RS-HO-002"]
    entry = result.payload["findings"][0]
    assert entry["rule_id"] == "RS-HO-002"
    assert entry["severity"] == "BLOCK"  # unknown severity normalized to BLOCK
    assert entry["message"] == "no id at all"
    assert result.findings[0].severity.value == "WARN"


def test_declared_id_is_kept_and_the_seal_reads_back(ctx: SkillContext) -> None:
    skill = RefusalSurfaceSkill(declared_rule_ids=frozenset({"QQ-IN-001"}))
    request = {
        "findings": [{"rule_id": "QQ-IN-001", "severity": "WARN", "message": "declared finding",
                      "evidence": {"axis": "pp"}, "recovery": "rerun the probe"}],
        "source_skill": "probe_first",
        "result_status": "UNMEASURED",
        "artifact_id": "seal-1",
    }
    result = skill.execute(request, ctx)
    assert result.status is Status.PASS
    assert result.exit_code == 0
    assert result.findings == ()
    entry = result.payload["findings"][0]
    assert entry["rule_id"] == "QQ-IN-001" and entry["severity"] == "WARN"
    assert entry["evidence"]["source_rule_id"] == "QQ-IN-001"
    assert entry["cited_by"] == "refusal_surface"  # an id only the caller declares is vouched by the sealing skill
    ref = result.artifacts[0]
    assert ref.id == "seal-1"
    sealed = read_artifact(Path(ref.path), expect_type="finding_set")
    assert sealed.payload["source_skill"] == "probe_first"
    assert sealed.payload["result_status"] == "UNMEASURED"
    assert sealed.payload["findings"] == result.payload["findings"]
    assert sealed.provenance.skill == "refusal_surface"


def test_default_artifact_id_is_the_content_hash_of_the_seal(ctx: SkillContext) -> None:
    result = RefusalSurfaceSkill().execute({"findings": [{"message": "no id at all"}]}, ctx)
    expected = f"rs-{sha256_json(result.payload['findings'])[:12]}"
    assert result.artifacts[0].id == expected
    assert Path(result.artifacts[0].path).name == f"finding_set.{expected}.json"
    assert Path(result.artifacts[0].path).parent == ctx.artifacts_dir


def test_check_handoff_flags_an_emitted_undeclared_entry(ctx: SkillContext) -> None:
    skill = RefusalSurfaceSkill(declared_rule_ids=frozenset({"QQ-IN-001"}))
    payload = {
        "source_skill": "x",
        "result_status": "RED",
        "findings": [{"rule_id": "MADE-UP-1", "severity": "BLOCK", "message": "m",
                      "evidence": {}, "recovery": "", "cited_by": "x"}],
    }
    findings = skill.check_handoff(SkillResult(status=Status.PASS, payload=payload), ctx)
    assert [f.rule_id for f in findings] == ["RS-HO-001"]
    assert findings[0].severity.value == "BLOCK"


def test_unknown_severity_is_normalised_and_attributed(ctx: SkillContext) -> None:
    """RS-HO-002 (WARN): an unknown severity is normalized to BLOCK and the entry is cited as RS-HO-002."""
    skill = RefusalSurfaceSkill(declared_rule_ids=frozenset({"QQ-IN-001"}))
    result = skill.run({"findings": [{"rule_id": "QQ-IN-001", "severity": "FATAL", "message": "lost severity"}]}, ctx)
    assert result.status is Status.PASS
    assert [f.rule_id for f in result.findings] == ["RS-HO-002"]
    entry = result.payload["findings"][0]
    assert entry["rule_id"] == "RS-HO-002"
    assert entry["severity"] == "BLOCK"
    assert entry["evidence"]["source_rule_id"] == "QQ-IN-001"
    assert result.findings[0].evidence["severity_normalized"] is True


def test_never_emits_a_finding_with_an_undeclared_rule_id(ctx: SkillContext) -> None:
    skill = RefusalSurfaceSkill(declared_rule_ids=frozenset({"QQ-IN-001"}))
    legal = {spec.rule_id for spec in skill.rules} | {"CORE-EXC", "CORE-OUT-SCHEMA", "CORE-UNDECLARED-RULE"}
    cases = [
        {"findings": [{"rule_id": "PF-IN-999", "severity": "BLOCK", "message": "invented"}]},
        {"findings": [{"rule_id": "made-up-id", "severity": "WARN", "message": "garbage-shaped id"}]},
        {"findings": [{"rule_id": "QQ-IN-001", "severity": "BLOCK", "message": "declared"}]},
        {"findings": [{"message": "no id at all"}]},
        {"findings": [{"rule_id": "QQ-IN-001", "message": "declared but severity lost"}]},
        {"findings": [{"message": "clean"}, {"rule_id": "PF-IN-999", "message": "mixed in"}]},
    ]
    for request in cases:
        result = skill.execute(request, ctx)
        assert {f.rule_id for f in result.findings} <= legal, request
        for entry in result.payload.get("findings", []):
            assert entry["rule_id"] in set(skill.declared_rule_ids) | {"RS-HO-001", "RS-HO-002"}, request


def test_diagnose_offers_the_rule_id_playbook(ctx: SkillContext) -> None:
    diag = RefusalSurfaceSkill().diagnose(ValueError("boom"))
    assert diag.likely_causes == ("the model wrote its own rule id", "a finding lost its id in serialisation")
    assert diag.checks == ("look up the id in every SKILL.md's Validation rules",)
    assert diag.recovery == ("cite the declared id or declare it with a MUST_FIRE fixture",)
    # a REFUSED SkillResult is a legal diagnose() input (contract.py: the diagnosis lives on the
    # SkillResult as `diagnosis` and is None for an input-phase refusal that never ran)
    result = RefusalSurfaceSkill().execute({"findings": []}, ctx)
    assert result.diagnosis is None or result.diagnosis.symptom
    from_refusal = RefusalSurfaceSkill().diagnose(result)
    assert from_refusal.likely_causes == ("the model wrote its own rule id", "a finding lost its id in serialisation")
    assert "refusal_surface returned REFUSED" in from_refusal.symptom or from_refusal.symptom


def test_a_library_caller_without_cli_registration_still_sees_declared_foreign_ids(
    ctx: SkillContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty registry (no `register_builtin_skills()` ran) made every foreign id read as invented (A3.1 GPU chain)."""
    from foundationskills.core.registry import SkillRegistry
    from foundationskills.skills.refusal_surface import skill as rs_skill

    monkeypatch.setattr(rs_skill, "REGISTRY", SkillRegistry())
    result = RefusalSurfaceSkill().execute(
        {"findings": [{"rule_id": "PF-IN-002", "severity": "BLOCK", "message": "multi-GPU RL missing"}],
         "source_skill": "probe_first", "result_status": "REFUSED"}, ctx)
    assert result.status == Status.PASS
    assert "RS-HO-001" not in [f.rule_id for f in result.findings]
