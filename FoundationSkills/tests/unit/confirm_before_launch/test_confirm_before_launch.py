"""MUST_FIRE matrix for ConfirmBeforeLaunchSkill (stdlib + pytest only, no GPU, no network).

The fixtures reference ``launch/fs_launch_spec.q27.json`` and ``missing.json``; each test
materialises the readable ones inside ``tmp_path`` and points ``SkillContext.workdir`` there, so
nothing is ever written outside the test sandbox.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import pytest

from foundationskills.core.contract import SkillContext, SkillResult
from foundationskills.core.orchestrator import plan_hash
from foundationskills.core.provenance import sha256_file
from foundationskills.core.status import Status
from foundationskills.skills.confirm_before_launch.skill import ConfirmBeforeLaunchSkill

SPEC_PAYLOAD: dict[str, Any] = {"entry": "fskills-eval", "executable": True, "run_name": "q27"}
SPEC_REL = "launch/fs_launch_spec.q27.json"
_CONFIRMED = plan_hash(SPEC_PAYLOAD)  # exactly what `fskills hash --spec` prints (16 hex)
_RULE_TOKEN = re.compile(r"\b[A-Z]{2,4}-[A-Z0-9_-]*\d\b")


def write_spec(tmp_path: Path, payload: dict[str, Any] | None = None) -> Path:
    target = tmp_path / SPEC_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload if payload is not None else SPEC_PAYLOAD, sort_keys=True), encoding="utf-8")
    return target


def records(tmp_path: Path) -> list[Path]:
    return sorted(p for p in tmp_path.rglob("confirmation_record*.json"))


def tampering_plan_hash(real: Callable[[dict[str, Any]], str], spec: Path) -> Callable[[dict[str, Any]], str]:
    """Rewrite the spec right after the first (input) recompute: drift before the record seal."""
    calls = {"n": 0}

    def wrapper(payload: dict[str, Any]) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            spec.write_text(json.dumps({**payload, "run_name": "q27-tampered"}, sort_keys=True), encoding="utf-8")
        return real(payload)

    return wrapper


def test_every_declared_rule_has_exactly_one_must_fire_fixture() -> None:
    skill = ConfirmBeforeLaunchSkill()
    fixtures = skill.must_fire_fixtures()
    assert set(fixtures) == {rule.rule_id for rule in skill.rules}
    for rule_id, fixture in fixtures.items():
        assert "request" in fixture and "expected" in fixture, rule_id
        assert fixtures == skill.must_fire_fixtures()  # deterministic


def test_cb_in_001_absent_user_plan_hash_refuses_without_echoing_any_hash(tmp_path: Path) -> None:
    """MUST_FIRE row CB-IN-001: `{request: {launch_spec: "launch/fs_launch_spec.q27.json"}}`."""
    spec = write_spec(tmp_path)
    ctx = SkillContext(workdir=tmp_path)
    result = ConfirmBeforeLaunchSkill().execute({"launch_spec": str(spec)}, ctx)
    assert result.status is Status.REFUSED
    assert result.exit_code == 96
    assert "CB-IN-001" in (result.refusal or "")
    assert records(tmp_path) == []
    dumped = json.dumps(result.to_dict())
    assert plan_hash(SPEC_PAYLOAD) not in dumped  # the 16-hex plan hash must never leave the gate
    assert sha256_file(spec) not in dumped


def test_cb_in_002_mismatching_user_plan_hash_refuses_without_echoing_any_hash(tmp_path: Path) -> None:
    """MUST_FIRE row CB-IN-002: `user_plan_hash: "0" * 16`."""
    spec = write_spec(tmp_path)
    result = ConfirmBeforeLaunchSkill().execute(
        {"launch_spec": str(spec), "user_plan_hash": "0" * 16}, SkillContext(workdir=tmp_path)
    )
    assert result.status is Status.REFUSED and result.exit_code == 96
    assert "CB-IN-002" in (result.refusal or "")
    assert "CB-IN-001" not in (result.refusal or "")
    assert records(tmp_path) == []
    assert plan_hash(SPEC_PAYLOAD) not in json.dumps(result.to_dict())


def test_cb_in_003_covers_missing_unreadable_and_foreign_launch_specs(tmp_path: Path) -> None:
    """MUST_FIRE row CB-IN-003 (missing.json) plus the rest of its condition."""
    skill = ConfirmBeforeLaunchSkill()
    ctx = SkillContext(workdir=tmp_path)
    non_object = tmp_path / "launch" / "non_object.json"
    non_object.parent.mkdir(parents=True, exist_ok=True)
    non_object.write_text("[]", encoding="utf-8")
    foreign = tmp_path / "launch" / "eval_report.q27.json"
    foreign.write_text(
        json.dumps({"type": "eval_report", "id": "q27", "payload": {"verdict": "PASS"}, "provenance": {}}),
        encoding="utf-8",
    )
    for raw in (str(tmp_path / "missing.json"), str(non_object), str(foreign)):
        result = skill.execute({"launch_spec": raw, "user_plan_hash": "f" * 16}, ctx)
        assert result.status is Status.REFUSED and result.exit_code == 96, raw
        assert "CB-IN-003" in (result.refusal or ""), raw
        assert records(tmp_path) == []


def test_an_empty_user_plan_hash_reaches_check_inputs_and_names_cb_in_001(tmp_path: Path) -> None:
    """`user_plan_hash` has no minLength: an empty string must cite a rule, not explode the schema."""
    spec = write_spec(tmp_path)
    result = ConfirmBeforeLaunchSkill().execute(
        {"launch_spec": str(spec), "user_plan_hash": ""}, SkillContext(workdir=tmp_path)
    )
    assert result.status is Status.REFUSED
    assert not (result.refusal or "").startswith("invalid request")
    assert "CB-IN-001" in (result.refusal or "")
    assert records(tmp_path) == []


def test_cb_ho_001_the_spec_drifted_between_the_input_recompute_and_the_seal(tmp_path: Path) -> None:
    """MUST_FIRE row CB-HO-001 (tampered between recompute and seal): REFUSED 96, no record."""
    spec = write_spec(tmp_path)
    skill = ConfirmBeforeLaunchSkill(plan_hash=tampering_plan_hash(plan_hash, spec))
    result = skill.execute({"launch_spec": str(spec), "user_plan_hash": _CONFIRMED}, SkillContext(workdir=tmp_path))
    assert result.status is Status.REFUSED and result.exit_code == 96
    assert "CB-HO-001" in (result.refusal or "")
    assert records(tmp_path) == []
    assert plan_hash(SPEC_PAYLOAD) not in json.dumps(result.to_dict())


def test_cb_ho_001_check_handoff_re_verifies_the_sealed_spec_sha256(tmp_path: Path) -> None:
    spec = write_spec(tmp_path)
    skill = ConfirmBeforeLaunchSkill()
    ctx = SkillContext(workdir=tmp_path)
    result = skill.execute({"launch_spec": str(spec), "user_plan_hash": _CONFIRMED}, ctx)
    assert result.status is Status.PASS
    spec.write_text(json.dumps({**SPEC_PAYLOAD, "run_name": "re-emitted-q27"}, sort_keys=True), encoding="utf-8")
    findings = skill.check_handoff(result, ctx)
    assert [f.rule_id for f in findings] == ["CB-HO-001"]
    assert findings[0].severity.value == "BLOCK"


def test_a_post_seal_re_emit_is_caught_by_the_handoff_re_verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The record is sealed against spec_sha256; a rewrite after the seal is a CB-HO-001 handoff BLOCK."""
    import foundationskills.skills.confirm_before_launch.skill as skill_module

    spec = write_spec(tmp_path)
    real_write = skill_module.write_artifact

    def drifting_write(artifact: Any, directory: Any) -> Any:
        ref = real_write(artifact, directory)
        spec.write_text(json.dumps({**SPEC_PAYLOAD, "run_name": "re-emitted-q27"}, sort_keys=True), encoding="utf-8")
        return ref

    monkeypatch.setattr(skill_module, "write_artifact", drifting_write)
    result = ConfirmBeforeLaunchSkill().execute(
        {"launch_spec": str(spec), "user_plan_hash": _CONFIRMED}, SkillContext(workdir=tmp_path)
    )
    assert result.status is not Status.PASS  # the sealed record no longer stands
    assert [f.rule_id for f in result.findings] == ["CB-HO-001"]


def test_the_user_confirmed_hash_seals_a_confirmation_record(tmp_path: Path) -> None:
    spec = write_spec(tmp_path)
    ctx = SkillContext(workdir=tmp_path)
    result = ConfirmBeforeLaunchSkill().execute({"launch_spec": str(spec), "user_plan_hash": _CONFIRMED}, ctx)
    assert result.status is Status.PASS and result.exit_code == 0
    (ref,) = result.artifacts
    assert ref.type == "confirmation_record"
    assert ref.id == f"cb-{sha256_file(spec)[:12]}"  # default id is content-addressed
    record_path = ctx.artifacts_dir / f"confirmation_record.{ref.id}.json"
    assert Path(ref.path) == record_path and record_path.is_file()
    sealed = json.loads(record_path.read_text(encoding="utf-8"))["payload"]
    assert sealed == {
        "launch_spec": str(spec),
        "spec_sha256": sha256_file(spec),
        "confirmed_hash": _CONFIRMED,
        "confirmed_by": "user",
        "auto_filled": False,
    }
    assert result.payload["auto_filled"] is False
    assert ref.sha256 == sha256_file(record_path)


def test_out_and_artifact_id_control_where_the_record_lands(tmp_path: Path) -> None:
    spec = write_spec(tmp_path)
    out = tmp_path / "seals"
    result = ConfirmBeforeLaunchSkill().execute(
        {"launch_spec": str(spec), "user_plan_hash": _CONFIRMED, "out": str(out), "artifact_id": "cb-q27"},
        SkillContext(workdir=tmp_path),
    )
    assert result.status is Status.PASS
    assert (out / "confirmation_record.cb-q27.json").is_file()
    assert records(tmp_path) == [out / "confirmation_record.cb-q27.json"]


@pytest.mark.parametrize("rule_id", ["CB-IN-001", "CB-IN-002", "CB-IN-003", "CB-HO-001"])
def test_must_fire_fixtures_refuse_with_their_own_rule(rule_id: str, tmp_path: Path) -> None:
    """Each fixture's request REFUSES (96) naming its rule and writes no confirmation_record."""
    write_spec(tmp_path)  # the fixture's launch/fs_launch_spec.q27.json, materialised in tmp_path
    fixture = ConfirmBeforeLaunchSkill().must_fire_fixtures()[rule_id]
    request = dict(fixture["request"])
    supplied = str(request.get("user_plan_hash", ""))
    plan_hasher = plan_hash
    if supplied.startswith("<"):  # "<the hash the user read back>": substituted by the driver
        request["user_plan_hash"] = plan_hash(SPEC_PAYLOAD)
    if "tamper" in fixture:
        plan_hasher = tampering_plan_hash(plan_hash, tmp_path / SPEC_REL)
    result = ConfirmBeforeLaunchSkill(plan_hash=plan_hasher).execute(request, SkillContext(workdir=tmp_path))
    assert result.status is Status.REFUSED and result.exit_code == 96
    assert rule_id in (result.refusal or ""), fixture["expected"]
    assert records(tmp_path) == []
    assert plan_hash(SPEC_PAYLOAD) not in json.dumps(result.to_dict())


def test_diagnose_names_the_two_causes_and_the_recompute_check() -> None:
    diagnosis = ConfirmBeforeLaunchSkill().diagnose(
        SkillResult(status=Status.REFUSED, payload={}, refusal="CB-IN-001: no user-supplied hash")
    )
    assert diagnosis.likely_causes == ("no user-supplied hash", "the spec was re-emitted after the hash was shown")
    assert diagnosis.checks == ("`fskills hash --spec <spec>`",)
    assert diagnosis.recovery == ("obtain the hash from the user again",)


def test_docs_and_evals_cite_only_declared_rules() -> None:
    skill = ConfirmBeforeLaunchSkill()
    declared = {rule.rule_id for rule in skill.rules}
    root = Path(__file__).resolve().parents[3] / "foundationskills" / "skills" / "confirm_before_launch"
    skill_md = (root / "SKILL.md").read_text(encoding="utf-8")
    evals = (root / "evals" / "evals.json").read_text(encoding="utf-8")
    cited = set(_RULE_TOKEN.findall(skill_md)) | set(_RULE_TOKEN.findall(evals))
    assert cited <= declared, cited - declared  # no invented rule ids
    assert declared <= set(_RULE_TOKEN.findall(skill_md))  # SKILL.md lists every declared rule
    for section in ("## Purpose", "## When to use", "## Inputs", "## Outputs", "## Running it on GB200",
                    "## Scope", "## Validation rules", "## Failure handling", "## FS interface",
                    "## Worked examples"):
        assert section in skill_md
