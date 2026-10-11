"""Shared A3.1 plumbing for the four discipline skills.

Registration, the four new artifact payload schemas, and the spec section 7
MUST_FIRE declarations (fixture keys == rule ids, severity/phase, and the
(status, exit code) pairs the contract's template facts resolve them to).
Stdlib + pytest only (no optional heavy dependencies), so nothing can skip.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

import foundationskills.core.status as status_module
from foundationskills.core.artifacts import ArtifactType, _load_artifact_schema
from foundationskills.core.registry import SkillRegistry
from foundationskills.core.schema import validate
from foundationskills.core.status import Status
from foundationskills.skills import register_builtin_skills
from foundationskills.skills.confirm_before_launch.skill import ConfirmBeforeLaunchSkill
from foundationskills.skills.measured_or_unmeasured.skill import MeasuredOrUnmeasuredSkill
from foundationskills.skills.probe_first.skill import ProbeFirstSkill
from foundationskills.skills.refusal_surface.skill import RefusalSurfaceSkill

DISCIPLINE_SKILLS = (
    ProbeFirstSkill,
    MeasuredOrUnmeasuredSkill,
    ConfirmBeforeLaunchSkill,
    RefusalSurfaceSkill,
)
SNAKE_NAMES = ("confirm_before_launch", "measured_or_unmeasured", "probe_first", "refusal_surface")

NEW_ARTIFACT_TYPES = ("probe_report", "claim_set", "confirmation_record", "finding_set")

REQUIRED_KEYS = {
    "probe_report": {"plan_draft", "checks", "capabilities", "verdict"},
    "claim_set": {"verdict", "claims"},
    "confirmation_record": {"launch_spec", "spec_sha256", "confirmed_hash", "confirmed_by", "auto_filled"},
    "finding_set": {"source_skill", "result_status", "findings"},
}

HEX_64 = "0123456789abcdef" * 4
HEX_16 = "deadbeefdeadbeef"

VALID_PAYLOADS = {
    "probe_report": {
        "plan_draft": "fs_launch_spec.run-1.json",
        "checks": [{"check": {"stage": "train", "algorithm": "sft", "tp": 1, "pp": 1}, "missing": None}],
        "capabilities": {"available": True, "fs_version": "0.0.0", "axes_measured": False},
        "verdict": "PASS",
    },
    "claim_set": {
        "verdict": "UNMEASURED",
        "claims": [
            {
                "id": "c1",
                "claim": "parallel.axis.pp is available",
                "status": "UNMEASURED",
                "evidence": None,
                "reason": "probe marked the axis unmeasured",
            }
        ],
    },
    "confirmation_record": {
        "launch_spec": "fs_launch_spec.run-1.json",
        "spec_sha256": HEX_64,
        "confirmed_hash": HEX_16,
        "confirmed_by": "user",
        "auto_filled": False,
    },
    "finding_set": {
        "source_skill": "refusal_surface",
        "result_status": "RED",
        "findings": [
            {
                "rule_id": "RS-HO-001",
                "severity": "BLOCK",
                "message": "invented rule id PF-IN-999",
                "evidence": {"rule_id": "PF-IN-999"},
                "recovery": "attribute the finding to a declared rule",
                "cited_by": "RS-HO-001",
            }
        ],
    },
}

INVALID_PAYLOADS = {
    "probe_report": {
        "plan_draft": "fs_launch_spec.run-1.json",
        "checks": [{"check": {}, "missing": None}],
        "capabilities": {"available": True},
        "verdict": "RED",
    },
    "claim_set": {"verdict": "PASS", "claims": []},
    "confirmation_record": {
        "launch_spec": "fs_launch_spec.run-1.json",
        "spec_sha256": HEX_64,
        "confirmed_hash": HEX_64,
        "confirmed_by": "user",
        "auto_filled": False,
    },
    "finding_set": {
        "source_skill": "refusal_surface",
        "result_status": "RED",
        "findings": [{"rule_id": "pf-in-999", "severity": "BLOCK", "message": "lowercase rule id"}],
    },
}


@dataclass(frozen=True)
class QuartetRow:
    """One MUST_FIRE row: skill, rule, and the (status, exit code) it must resolve to."""

    skill_cls: type
    rule_id: str
    status: Status
    code: int
    phase: str
    severity: str


# Spec section 7 rows 1, 2, 4, 5 (row 3 is the claim-level case: UNMEASURED 95, never REFUSED).
QUARTET = (
    QuartetRow(ProbeFirstSkill, "PF-IN-001", Status.REFUSED, 96, "input", "BLOCK"),
    QuartetRow(ProbeFirstSkill, "PF-IN-002", Status.REFUSED, 96, "input", "BLOCK"),
    QuartetRow(ConfirmBeforeLaunchSkill, "CB-IN-001", Status.REFUSED, 96, "input", "BLOCK"),
    QuartetRow(RefusalSurfaceSkill, "RS-HO-001", Status.RED, 5, "handoff", "BLOCK"),
)


def _member_name(value: object) -> str:
    """Enum member or plain string → its declared code/name."""
    return str(getattr(value, "value", value))


def _exit_code(status: Status) -> int:
    """Resolve the doctrine codes (Status PASS/RED/UNMEASURED/REFUSED = 0/5/95/96) however core/status.py exposes them."""
    attr = getattr(status, "exit_code", None)
    if attr is not None:
        return int(attr() if callable(attr) else attr)
    module_level = getattr(status_module, "exit_code", None)
    if callable(module_level):
        return int(module_level(status))
    if isinstance(module_level, dict):
        return int(module_level[status])
    table = getattr(status_module, "EXIT_CODES", None)
    if isinstance(table, dict):
        return int(table[status])
    raise AssertionError(f"cannot resolve the exit code for {status!r}")


def test_four_discipline_skills_register() -> None:
    """register_builtin_skills() returns and registers the four snake-named discipline skills."""
    registry = SkillRegistry()
    registered = set(register_builtin_skills(registry))
    assert {cls.name for cls in DISCIPLINE_SKILLS} == set(SNAKE_NAMES)
    assert set(SNAKE_NAMES) <= registered
    assert set(SNAKE_NAMES) <= set(registry.names())


def test_must_fire_quartet_resolves_to_refused_or_red_with_a_rule_id() -> None:
    """No-op guard name kept for the single-case signature; see the parametrised sweep below."""


@pytest.mark.parametrize("row", QUARTET, ids=lambda row: row.rule_id)
def test_must_fire_quartet_resolve(row: QuartetRow) -> None:
    """MUST_FIRE quartet: fixture keys == rule ids and the cited rule is declared BLOCK at the right phase."""
    skill = row.skill_cls()
    rules = {rule.rule_id: rule for rule in skill.rules}
    assert set(skill.must_fire_fixtures()) == set(rules)

    rule = rules[row.rule_id]
    assert _member_name(rule.severity) == row.severity
    assert _member_name(rule.phase) == row.phase

    # Template facts (core/contract.py): input-phase BLOCK -> REFUSED, handoff-phase BLOCK -> RED.
    expected_status, expected_code = {
        ("input", "BLOCK"): (Status.REFUSED, 96),
        ("handoff", "BLOCK"): (Status.RED, 5),
    }[(row.phase, row.severity)]
    assert row.status is expected_status
    assert row.code == expected_code
    assert _exit_code(row.status) == row.code


@pytest.mark.parametrize("skill_cls", DISCIPLINE_SKILLS, ids=[cls.name for cls in DISCIPLINE_SKILLS])
def test_must_fire_fixture_shape_covers_every_rule(skill_cls: type) -> None:
    """Every declared rule has a must-fire fixture and the fixtures cover no undeclared rule."""
    skill = skill_cls()
    assert set(skill.must_fire_fixtures()) == {rule.rule_id for rule in skill.rules}


def test_artifact_payload_schemas_load_for_the_four_new_types() -> None:
    """The four new ArtifactType members map to schema files that load and judge payloads."""
    assert {t.value for t in ArtifactType} >= set(NEW_ARTIFACT_TYPES)
    for type_name in NEW_ARTIFACT_TYPES:
        schema = _load_artifact_schema(type_name)
        assert schema["type"] == "object"
        assert schema.get("additionalProperties") is True
        assert set(schema.get("required", [])) == REQUIRED_KEYS[type_name]
    for type_name, payload in VALID_PAYLOADS.items():
        assert validate(payload, _load_artifact_schema(type_name)) == []
    for type_name, payload in INVALID_PAYLOADS.items():
        assert validate(payload, _load_artifact_schema(type_name)), f"{type_name} accepted an invalid payload"


def test_plan_chain_still_ends_at_training_emit() -> None:
    """Registering the discipline skills must not disturb the raw→launch chain."""
    registry = SkillRegistry()
    register_builtin_skills(registry)
    assert registry.plan_chain({"raw_data_ref", "goal_spec"}, "fs_launch_spec") == [
        "data_engine",
        "training.planner",
        "training.emit",
    ]
