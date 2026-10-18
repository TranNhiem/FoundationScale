"""A3.2 composition helper for user-invoked workflow skills.

A workflow skill (``invocation = "user"``) runs model skills it declares in ``invokes``. Their findings carry rule ids
the workflow skill does not declare, and an undeclared rule id turns any result RED (``BaseSkill.execute``), so the
inner findings travel as EVIDENCE of the workflow skill's own declared rule, never as its findings. The inner status
maps onto the workflow skill's rules by outcome class:

    inner REFUSED     -> the caller's input-phase BLOCK rule  -> REFUSED (96)
    inner RED         -> the caller's handoff-phase BLOCK rule -> RED (5)
    inner UNMEASURED  -> the caller's WARN rule               -> UNMEASURED (95), never PASS
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from foundationskills.core.contract import SkillContext, SkillResult
from foundationskills.core.registry import SkillRegistry
from foundationskills.core.status import Status


@dataclass
class Step:
    """One composed model-skill run: what ran, how it ended, and what it wrote."""

    skill: str
    status: str
    exit_code: int
    refusal: str | None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {"skill": self.skill, "status": self.status, "exit_code": self.exit_code, "refusal": self.refusal,
                "artifacts": list(self.artifacts), "findings": list(self.findings)}


def registry() -> SkillRegistry:
    """A fresh registry of every built-in skill (imported lazily: skill modules import this helper)."""
    from foundationskills.skills import register_builtin_skills  # noqa: PLC0415

    fresh = SkillRegistry()
    register_builtin_skills(fresh)
    return fresh


def run_step(caller: Any, name: str, request: dict[str, Any], ctx: SkillContext,
             reg: SkillRegistry | None = None) -> tuple[SkillResult, Step]:
    """Run model skill ``name`` for workflow skill ``caller``; refuses a skill the caller does not declare."""
    if name not in tuple(getattr(caller, "invokes", ())):
        raise ValueError(f"{caller.name} composes {name!r}, which it does not declare in invokes")
    result = (reg or registry()).get(name).execute(request, ctx)
    step = Step(
        skill=name,
        status=result.status.value,
        exit_code=int(result.exit_code),
        refusal=result.refusal,
        artifacts=[ref.to_dict() for ref in result.artifacts],
        findings=[{"rule_id": f.rule_id, "severity": f.severity.value, "message": f.message, "recovery": f.recovery}
                  for f in result.findings],
    )
    return result, step


def outcome_class(result: SkillResult) -> str:
    """"ok" | "refused" | "red" | "unmeasured", the class the caller maps onto its own declared rule."""
    return {Status.PASS: "ok", Status.REFUSED: "refused", Status.RED: "red",
            Status.UNMEASURED: "unmeasured"}[result.status]


def load_payload(path: str) -> dict[str, Any]:
    """Read a JSON file and unwrap an artifact envelope to its payload, exactly as the fskills CLI does."""
    import json  # noqa: PLC0415

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    if "payload" in data and "type" in data and isinstance(data["payload"], dict):
        return dict(data["payload"])
    return data


def artifact_paths(result: SkillResult, kind: str) -> list[str]:
    """Paths of the artifacts of type ``kind`` a composed step wrote (outputs reference artifacts by path)."""
    return [str(ref.path) for ref in result.artifacts if str(ref.type) == kind]
