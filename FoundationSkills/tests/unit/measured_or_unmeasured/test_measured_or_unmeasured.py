"""MUST_FIRE matrix for MeasuredOrUnmeasuredSkill (pure stdlib rewrite over evidence pointers)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foundationskills.core.artifacts import read_artifact
from foundationskills.core.contract import SkillContext
from foundationskills.core.provenance import sha256_json
from foundationskills.core.status import Status
from foundationskills.skills.measured_or_unmeasured.evidence import (
    EvidenceError,
    resolve_evidence,
)
from foundationskills.skills.measured_or_unmeasured.skill import MeasuredOrUnmeasuredSkill

EVAL_REPORT = {
    "verdict": "PASS",
    "benchmarks": [{"name": "mmlu", "metric": "acc,none", "score": 0.52, "status": "pass"}],
}


def write_eval_report(tmp_path: Path) -> str:
    """The artifact file the fixtures reference — created here, under tmp_path, by the test itself."""
    path = tmp_path / "eval_report.json"
    path.write_text(json.dumps(EVAL_REPORT), encoding="utf-8")
    return str(path)


def ctx(tmp_path: Path) -> SkillContext:
    return SkillContext(workdir=tmp_path)


def rule_ids(result) -> set[str]:
    return {f.rule_id for f in result.findings}


def test_must_fire_fixtures_cover_every_rule() -> None:
    skill = MeasuredOrUnmeasuredSkill()
    assert set(skill.must_fire_fixtures()) == {rule.rule_id for rule in skill.rules}


def test_all_rules_are_warn_input_so_the_exit_set_stays_0_95() -> None:
    for rule in MeasuredOrUnmeasuredSkill().rules:
        assert rule.severity.value == "WARN"
        assert rule.phase == "input"


@pytest.mark.parametrize("rule_id", ["MU-IN-001", "MU-IN-002", "MU-IN-003"])
def test_must_fire_yields_unmeasured_95_and_never_pass(tmp_path: Path, rule_id: str) -> None:
    fixture = MeasuredOrUnmeasuredSkill().must_fire_fixtures()[rule_id]
    result = MeasuredOrUnmeasuredSkill().execute(fixture["request"], ctx(tmp_path))
    assert rule_id in rule_ids(result)
    assert result.status is Status.UNMEASURED
    assert result.status is not Status.PASS
    assert result.exit_code == 95


def test_mu_in_001_pass_without_evidence_is_never_a_silent_pass(tmp_path: Path) -> None:
    fixture = MeasuredOrUnmeasuredSkill().must_fire_fixtures()["MU-IN-001"]
    result = MeasuredOrUnmeasuredSkill().execute(fixture["request"], ctx(tmp_path))
    assert result.status is Status.UNMEASURED
    assert result.exit_code == 95
    assert result.refusal is None
    assert "MU-IN-001" in rule_ids(result)
    assert result.payload["verdict"] == "UNMEASURED"
    assert result.payload["claims"][0]["id"] == "c1"
    assert result.payload["claims"][0]["status"] == "UNMEASURED"  # written UNMEASURED, never PASS
    assert result.payload["claims"][0]["reason"].startswith("UNMEASURED: ")
    sealed = json.loads(Path(result.artifacts[0].path).read_text(encoding="utf-8"))
    assert sealed["payload"]["claims"][0]["status"] == "UNMEASURED"
    assert sealed["payload"]["verdict"] == "UNMEASURED"
    assert Path(result.artifacts[0].path).is_relative_to(tmp_path)  # nothing written outside tmp_path


def test_mu_in_002_malformed_pointer_demotes_the_claim(tmp_path: Path) -> None:
    fixture = MeasuredOrUnmeasuredSkill().must_fire_fixtures()["MU-IN-002"]
    result = MeasuredOrUnmeasuredSkill().execute(fixture["request"], ctx(tmp_path))
    assert result.status is Status.UNMEASURED
    assert result.exit_code == 95
    assert "MU-IN-002" in rule_ids(result)
    assert result.payload["claims"][0]["status"] == "UNMEASURED"


def test_mu_in_002_pointer_to_a_file_not_on_disk_demotes_the_claim(tmp_path: Path) -> None:
    request = {"claims": [{"id": "c1", "claim": "sft holds general ability", "asserted_status": "PASS",
                           "evidence": f"{tmp_path / 'absent_report.json'}#benchmarks.0.score"}]}
    result = MeasuredOrUnmeasuredSkill().execute(request, ctx(tmp_path))
    assert result.status is Status.UNMEASURED
    assert result.exit_code == 95
    assert "MU-IN-002" in rule_ids(result)
    assert result.payload["claims"][0]["status"] == "UNMEASURED"
    assert not (tmp_path / "absent_report.json").exists()  # the skill only ever reads


def test_mu_in_002_a_null_score_on_disk_is_not_a_measurement(tmp_path: Path) -> None:
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps({"benchmarks": [{"score": None}]}), encoding="utf-8")
    request = {"claims": [{"id": "c1", "claim": "sft holds general ability", "asserted_status": "PASS",
                           "evidence": f"{path}#benchmarks.0.score"}]}
    result = MeasuredOrUnmeasuredSkill().execute(request, ctx(tmp_path))
    assert result.status is Status.UNMEASURED
    assert "MU-IN-002" in rule_ids(result)
    assert result.payload["claims"][0]["status"] == "UNMEASURED"


def test_mu_in_003_no_verdict_at_all_demotes_the_claim(tmp_path: Path) -> None:
    fixture = MeasuredOrUnmeasuredSkill().must_fire_fixtures()["MU-IN-003"]
    result = MeasuredOrUnmeasuredSkill().execute(fixture["request"], ctx(tmp_path))
    assert result.status is Status.UNMEASURED
    assert result.exit_code == 95
    assert "MU-IN-003" in rule_ids(result)
    assert result.payload["claims"][0]["status"] == "UNMEASURED"


def test_every_claim_measured_is_the_only_way_to_pass(tmp_path: Path) -> None:
    report_path = write_eval_report(tmp_path)  # created in tmp_path by this test
    request = {"claims": [{"id": "c1", "claim": "sft holds general ability", "asserted_status": "PASS",
                           "evidence": f"{report_path}#benchmarks.0.score"}]}
    skill = MeasuredOrUnmeasuredSkill()
    result = skill.execute(request, ctx(tmp_path))
    assert result.status is Status.PASS
    assert result.exit_code == 0
    assert result.payload["verdict"] == "PASS"
    assert result.payload["claims"][0]["status"] == "MEASURED"
    assert result.findings == ()
    assert skill.check_handoff(result, ctx(tmp_path)) == []  # BLOCK-free guard


def test_worst_holds_the_claim_set_at_95_with_one_unevidenced_claim(tmp_path: Path) -> None:
    report_path = write_eval_report(tmp_path)
    request = {"claims": [
        {"id": "c1", "claim": "measured claim", "asserted_status": "PASS",
         "evidence": f"{report_path}#benchmarks.0.score"},
        {"id": "c2", "claim": "asserted claim", "asserted_status": "PASS"},
        {"id": "c3", "claim": "claim with no verdict at all"},
    ]}
    result = MeasuredOrUnmeasuredSkill().execute(request, ctx(tmp_path))
    assert result.status is Status.UNMEASURED
    assert result.exit_code == 95
    assert result.payload["verdict"] == "UNMEASURED"
    assert {r["id"]: r["status"] for r in result.payload["claims"]} == {
        "c1": "MEASURED", "c2": "UNMEASURED", "c3": "UNMEASURED"}
    assert rule_ids(result) == {"MU-IN-001", "MU-IN-003"}


def test_sealed_claim_set_round_trips_through_read_artifact(tmp_path: Path) -> None:
    request = {"claims": [{"id": "c1", "claim": "sft holds general ability", "asserted_status": "PASS"}]}
    result = MeasuredOrUnmeasuredSkill().execute(request, ctx(tmp_path))
    artifact = read_artifact(result.artifacts[0].path, expect_type="claim_set")
    assert artifact.type == "claim_set"
    assert artifact.payload["verdict"] == "UNMEASURED"
    assert artifact.payload["claims"][0]["status"] == "UNMEASURED"
    assert artifact.provenance.skill == "measured_or_unmeasured"


def test_default_artifact_id_is_the_claims_hash(tmp_path: Path) -> None:
    request = {"claims": [{"id": "c1", "claim": "sft holds general ability"}]}
    result = MeasuredOrUnmeasuredSkill().execute(request, ctx(tmp_path))
    expected_id = f"mu-{sha256_json(request['claims'])[:12]}"
    assert result.artifacts[0].id == expected_id
    assert Path(result.artifacts[0].path) == tmp_path / "artifacts" / f"claim_set.{expected_id}.json"


def test_out_directory_and_explicit_artifact_id(tmp_path: Path) -> None:
    request = {"claims": [{"id": "c1", "claim": "sft holds general ability"}],
               "out": str(tmp_path / "sealed"), "artifact_id": "mu-custom"}
    result = MeasuredOrUnmeasuredSkill().execute(request, ctx(tmp_path))
    assert Path(result.artifacts[0].path) == tmp_path / "sealed" / "claim_set.mu-custom.json"
    assert Path(result.artifacts[0].path).is_relative_to(tmp_path)


def test_invalid_request_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    result = MeasuredOrUnmeasuredSkill().execute({"claims": []}, ctx(tmp_path))
    assert result.status is Status.REFUSED
    assert result.exit_code == 96
    assert result.refusal and result.refusal.startswith("invalid request")
    assert result.artifacts == ()
    assert list(tmp_path.iterdir()) == []  # no report, no artifact, nothing


def test_diagnose_names_the_two_documented_causes() -> None:
    diagnosis = MeasuredOrUnmeasuredSkill().diagnose(RuntimeError("boom"))
    assert "claim was asserted without evidence" in diagnosis.likely_causes
    assert "pointer names a file that is not on disk" in diagnosis.likely_causes
    assert "read the pointer's `#field`" in diagnosis.checks
    assert "measure it, then re-run" in diagnosis.recovery


def test_injected_resolver_decides_measured_versus_demoted(tmp_path: Path) -> None:
    resolved: list[str] = []

    def fake_resolve(pointer: str) -> Any:
        resolved.append(pointer)
        if pointer == "ok:value":
            return 0.5
        raise EvidenceError(f"cannot resolve {pointer!r}")

    skill = MeasuredOrUnmeasuredSkill(resolve=fake_resolve)
    request = {"claims": [
        {"id": "c1", "claim": "measured claim", "asserted_status": "PASS", "evidence": "ok:value"},
        {"id": "c2", "claim": "asserted claim", "asserted_status": "PASS", "evidence": "nope:value"},
    ]}
    result = skill.execute(request, ctx(tmp_path))
    assert result.status is Status.UNMEASURED
    assert {r["id"]: r["status"] for r in result.payload["claims"]} == {"c1": "MEASURED", "c2": "UNMEASURED"}
    # deterministic double pass: one resolution per claim in check_inputs, then again in run
    assert resolved == ["ok:value", "nope:value", "ok:value", "nope:value"]


def test_resolve_evidence_walks_dotted_fields(tmp_path: Path) -> None:
    path = tmp_path / "eval_report.json"
    path.write_text(json.dumps(EVAL_REPORT), encoding="utf-8")
    assert resolve_evidence(f"{path}#benchmarks.0.score") == 0.52
    assert resolve_evidence(f"{path}#verdict") == "PASS"


def test_resolve_evidence_rejects_malformed_and_unresolvable_pointers(tmp_path: Path) -> None:
    path = tmp_path / "eval_report.json"
    path.write_text(json.dumps(EVAL_REPORT), encoding="utf-8")
    with pytest.raises(EvidenceError):
        resolve_evidence("we ran it last week")  # not <path>#<field>
    with pytest.raises(EvidenceError):
        resolve_evidence(f"{path}#")  # no field
    with pytest.raises(EvidenceError):
        resolve_evidence(f"{tmp_path / 'absent.json'}#verdict")  # file not on disk
    with pytest.raises(EvidenceError):
        resolve_evidence(f"{path}#benchmarks.0.stderr")  # field does not resolve


def test_a_resolved_pointer_without_asserted_status_is_measured_and_not_flagged(tmp_path: Path) -> None:
    """MU-IN-003 said 'emitted UNMEASURED' on a claim written MEASURED (A3.1 GPU chain, real eval reports)."""
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"benchmarks": [{"score": 0.54}]}), encoding="utf-8")
    request = {"claims": [{"id": "c1", "claim": "sft scores on arc_easy", "evidence": f"{path}#benchmarks.0.score"}]}
    result = MeasuredOrUnmeasuredSkill().execute(request, ctx(tmp_path))
    assert result.status is Status.PASS
    assert result.payload["claims"][0]["status"] == "MEASURED"
    assert "MU-IN-003" not in rule_ids(result)
