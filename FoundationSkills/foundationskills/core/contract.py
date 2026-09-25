"""Skill contract datatypes and the concrete execute template."""
from __future__ import annotations

import abc
import re
from enum import Enum
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from foundationskills.core.artifacts import ArtifactRef
from foundationskills.core.provenance import Provenance
from foundationskills.core.schema import assert_valid, SchemaError, validate
from foundationskills.core.status import Status


class Severity(str, Enum):
    BLOCK = "BLOCK"
    WARN = "WARN"
    INFO = "INFO"


_RULE_ID = re.compile(r"^[A-Z]{2,4}-[A-Z0-9_-]+$")


@dataclass(frozen=True)
class RuleSpec:
    """Declared validation rule; every rule needs a must-fire negative fixture."""

    rule_id: str
    description: str
    severity: Severity
    phase: Literal["input", "handoff"]

    def __post_init__(self) -> None:
        if not _RULE_ID.fullmatch(self.rule_id):
            raise ValueError(f"invalid rule_id {self.rule_id!r}; expected ^[A-Z]{{2,4}}-[A-Z0-9_-]+$")
        object.__setattr__(self, "severity", Severity(self.severity))


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: Severity
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    recovery: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", Severity(self.severity))


@dataclass(frozen=True)
class Scope:
    model_types: tuple[str, ...]
    families: tuple[str, ...] | Literal["any"]
    stages: tuple[str, ...]
    algorithms: tuple[str, ...]
    methods: tuple[str, ...]
    hardware: tuple[str, ...]

    def __post_init__(self) -> None:
        bad = [m for m in self.model_types if m not in {"llm", "vlm"}]
        if bad:
            raise ValueError(f"unknown model_types: {bad!r}")
        object.__setattr__(self, "model_types", tuple(self.model_types))
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "algorithms", tuple(self.algorithms))
        object.__setattr__(self, "methods", tuple(self.methods))
        object.__setattr__(self, "hardware", tuple(self.hardware))
        if self.families != "any":
            object.__setattr__(self, "families", tuple(self.families))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_types": list(self.model_types),
            "families": "any" if self.families == "any" else list(self.families),
            "stages": list(self.stages),
            "algorithms": list(self.algorithms),
            "methods": list(self.methods),
            "hardware": list(self.hardware),
        }


@dataclass(frozen=True)
class Diagnosis:
    symptom: str
    likely_causes: tuple[str, ...]
    checks: tuple[str, ...]
    recovery: tuple[str, ...]


@dataclass(frozen=True)
class FSInterface:
    entries: tuple[str, ...]
    emits: tuple[str, ...]
    apis: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class SkillContext:
    workdir: Path
    capabilities: Any = None
    dry_run: bool = False
    env: Mapping[str, str] = field(default_factory=dict)

    @property
    def artifacts_dir(self) -> Path:
        return self.workdir / "artifacts"


@dataclass(frozen=True)
class SkillResult:
    status: Status
    payload: dict[str, Any]
    artifacts: tuple[ArtifactRef, ...] = ()
    findings: tuple[Finding, ...] = ()
    refusal: str | None = None
    next_actions: tuple[str, ...] = ()
    diagnosis: Diagnosis | None = None
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        status = Status(self.status)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "next_actions", tuple(self.next_actions))
        if status is Status.REFUSED and not (self.refusal and self.refusal.strip()):
            raise ValueError("REFUSED requires a non-empty refusal naming the missing input/capability")
        if status is not Status.REFUSED and self.refusal is not None:
            raise ValueError("non-REFUSED results must have refusal None")
        if status is Status.PASS and any(f.severity is Severity.BLOCK for f in self.findings):
            raise ValueError("PASS must not carry BLOCK findings")

    @property
    def exit_code(self) -> int:
        return self.status.exit_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "exit_code": self.exit_code,
            "payload": self.payload,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "findings": [
                {
                    "rule_id": f.rule_id,
                    "severity": f.severity.value,
                    "message": f.message,
                    "evidence": f.evidence,
                    "recovery": f.recovery,
                }
                for f in self.findings
            ],
            "refusal": self.refusal,
            "next_actions": list(self.next_actions),
            "diagnosis": None
            if self.diagnosis is None
            else {
                "symptom": self.diagnosis.symptom,
                "likely_causes": list(self.diagnosis.likely_causes),
                "checks": list(self.diagnosis.checks),
                "recovery": list(self.diagnosis.recovery),
            },
            "provenance": None if self.provenance is None else self.provenance.to_dict(),
        }


class BaseSkill(abc.ABC):
    """Base class for skills. Subclasses define schemas/rules and implement hooks."""

    name: str = ""
    version: str = ""
    description: str = ""
    scope: Scope = Scope((), (), (), (), (), ())
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    input_schema: dict[str, Any] = {"type": "object"}
    output_schema: dict[str, Any] = {"type": "object"}
    rules: tuple[RuleSpec, ...] = ()
    fs_interface: FSInterface = FSInterface((), (), ())

    _CORE_RULES = {"CORE-EXC", "CORE-OUT-SCHEMA", "CORE-UNDECLARED-RULE"}

    @abc.abstractmethod
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        """Return findings for required inputs/capabilities before execution."""

    @abc.abstractmethod
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Execute the measured work and return a SkillResult."""

    @abc.abstractmethod
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        """Validate a PASS result before it is handed to downstream skills."""

    @abc.abstractmethod
    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        """Explain a failure and propose checks/recovery steps."""

    @abc.abstractmethod
    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        """Return one negative fixture per declared rule keyed by rule_id."""

    def rule(self, rule_id: str) -> RuleSpec:
        for spec in self.rules:
            if spec.rule_id == rule_id:
                return spec
        raise KeyError(f"undeclared rule {rule_id!r} for skill {self.name!r}")

    def finding(self, rule_id: str, message: str, evidence: dict[str, Any] | None = None, recovery: str = "") -> Finding:
        """Build a finding using the declared severity; KeyError if undeclared."""
        spec = self.rule(rule_id)
        return Finding(spec.rule_id, spec.severity, message, dict(evidence or {}), recovery)

    def _refused(self, reason: str, findings: Sequence[Finding] = ()) -> SkillResult:
        return SkillResult(status=Status.REFUSED, payload={}, findings=tuple(findings), refusal=reason)

    def execute(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        """Template: schema, input checks, run, output schema, handoff, rule declaration audit."""
        try:
            assert_valid(request, self.input_schema, f"{self.name} request")
        except SchemaError as exc:
            first = str(exc).split("; ", 1)[0]
            return self._refused(f"invalid request: {first}")

        input_findings = list(self.check_inputs(request, ctx) or [])
        blockers = [f for f in input_findings if f.severity is Severity.BLOCK]
        if blockers:
            refusal = "; ".join(f"{f.rule_id}: {f.message}" for f in blockers)
            return self._refused(refusal, input_findings)

        try:
            result = self.run(request, ctx)
            if not isinstance(result, SkillResult):
                raise TypeError(f"run() must return SkillResult, got {type(result).__name__}")
        except Exception as exc:  # a measured runtime failure is RED; Ctrl-C/SystemExit propagate
            try:
                diagnosis = self.diagnose(exc)
            except Exception:  # noqa: BLE001 - a broken diagnose must not mask the original failure
                diagnosis = None
            finding = Finding(
                "CORE-EXC",
                Severity.BLOCK,
                f"exception in {self.name}.run: {type(exc).__name__}: {exc}",
                {"exception_type": type(exc).__name__},
                "inspect evidence and rerun with verbose logging",
            )
            return SkillResult(
                status=Status.RED, payload={}, findings=(*input_findings, finding), diagnosis=diagnosis
            )

        # Non-blocking input findings (WARN/INFO) travel with the result.
        findings = [*input_findings, *result.findings]
        status = result.status
        if status is Status.PASS:
            output_errors = validate(result.payload, self.output_schema)
            if output_errors:
                status = Status.RED
                findings.append(
                    Finding(
                        "CORE-OUT-SCHEMA",
                        Severity.BLOCK,
                        "run result violates output_schema",
                        {"errors": output_errors},
                        "fix the skill output payload or tighten output_schema with a migration",
                    )
                )
        if status is Status.PASS:
            handoff_findings = list(self.check_handoff(result, ctx) or [])
            findings.extend(handoff_findings)
            if any(f.severity is Severity.BLOCK for f in handoff_findings):
                status = Status.RED

        declared = {r.rule_id for r in self.rules}
        unknown = sorted({f.rule_id for f in findings if f.rule_id not in declared and f.rule_id not in self._CORE_RULES})
        if unknown:
            status = Status.RED if status is not Status.REFUSED else status
            findings.append(
                Finding(
                    "CORE-UNDECLARED-RULE",
                    Severity.BLOCK,
                    f"undeclared rule findings emitted: {', '.join(unknown)}",
                    {"unknown_rule_ids": unknown},
                    "declare every rule in BaseSkill.rules and add must-fire fixtures",
                )
            )

        if status is Status.REFUSED:
            return replace(result, status=status, findings=tuple(findings))
        return replace(result, status=status, refusal=None, findings=tuple(findings))
