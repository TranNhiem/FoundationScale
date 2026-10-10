"""RefusalSurfaceSkill: attribute RED/REFUSED findings to declared rule ids and seal a finding_set.

Exit contract: 96 REFUSED when there is nothing to surface (an empty or non-finding
list -- no artifact is written); 5 RED when an incoming finding cites a rule id that
no registered skill declares (an invented/undeclared id -> RS-HO-001); 0 PASS
otherwise, with every unattributed or malformed finding attributed in the emitted
set as RS-HO-002 (WARN, an unknown severity is normalized to BLOCK) instead of
dropped. A Finding is never emitted with an undeclared rule id -- invented ids live
only in Finding.evidence and in the sealed payload (evidence.source_rule_id).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from foundationskills.core.artifacts import Artifact, read_artifact, write_artifact
from foundationskills.core.contract import (
    BaseSkill,
    Diagnosis,
    Finding,
    FSInterface,
    RuleSpec,
    Scope,
    Severity,
    SkillContext,
    SkillResult,
)
from foundationskills.core.provenance import make_provenance, sha256_json
from foundationskills.core.registry import REGISTRY
from foundationskills.core.status import Status

# core/contract.py:23 -- the RuleSpec/Finding rule-id pattern this skill honours.
_RULE_ID = re.compile(r"^[A-Z]{2,4}-[A-Z0-9_-]+$")
_SEVERITIES = ("BLOCK", "WARN", "INFO")


class RefusalSurfaceSkill(BaseSkill):
    name = "refusal_surface"
    version = "0.1.0"
    description = (
        "Attribute RED/REFUSED findings to rule ids a registered skill actually declares and seal the result "
        "as a finding_set (RS-HO-001 for invented ids, RS-HO-002 for unattributed or malformed findings)."
    )
    scope = Scope(
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "dgx-h100", "local"),
    )
    consumes = ("finding_set",)
    produces = ("finding_set",)

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["findings"],
        "additionalProperties": False,
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["message"],
                    "properties": {
                        "rule_id": {"type": "string"},
                        "severity": {"enum": ["BLOCK", "WARN", "INFO"]},
                        "message": {"type": "string", "minLength": 1},
                        "evidence": {"type": "object"},
                        "recovery": {"type": "string"},
                    },
                },
            },
            "source_skill": {"type": "string"},
            "result_status": {"enum": ["PASS", "RED", "UNMEASURED", "REFUSED"]},
            "out": {"type": "string"},
            "artifact_id": {"type": "string"},
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "required": ["source_skill", "result_status", "findings"],
        "additionalProperties": True,
        "properties": {
            "source_skill": {"type": "string"},
            "result_status": {"enum": ["PASS", "RED", "UNMEASURED", "REFUSED"]},
            "findings": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["rule_id", "severity", "message"],
                    "properties": {
                        "rule_id": {"type": "string", "pattern": "^[A-Z]{2,4}-[A-Z0-9_-]+$"},
                        "severity": {"enum": ["BLOCK", "WARN", "INFO"]},
                        "message": {"type": "string"},
                        "evidence": {"type": "object"},
                        "recovery": {"type": "string"},
                        "cited_by": {"type": "string"},
                    },
                },
            },
        },
    }
    rules = (
        RuleSpec("RS-IN-001", "findings empty or not a finding set: nothing to surface (refuse rather than emit an empty surface)",
                 Severity.BLOCK, "input"),
        RuleSpec("RS-HO-001", "an incoming finding cites a rule id no registered skill declares (an invented id, e.g. PF-IN-999)",
                 Severity.BLOCK, "handoff"),
        RuleSpec("RS-HO-002", "an incoming finding is unattributed or malformed (no usable rule id, or an unknown severity)",
                 Severity.WARN, "handoff"),
    )
    fs_interface = FSInterface((), ("finding_set",), (), "attribution only; no FoundationScale call")

    def __init__(self, declared_rule_ids: frozenset[str] | None = None) -> None:
        """Optionally pin the declared-id set (e.g. for a skill under development).

        The default is the union of this skill's own rule ids with the rule ids of
        every skill registered in this install (looked up through the SkillRegistry
        after ``register_builtin_skills()`` has run).
        """
        self._declared_override = None if declared_rule_ids is None else frozenset(declared_rule_ids)

    # -- declared rule ids ----------------------------------------------------------------------
    @staticmethod
    def _registered_declared() -> dict[str, str]:
        """rule_id -> declaring skill name for every skill registered in this install."""
        # a library caller never ran the CLI's registration: without it every foreign id reads as invented
        from foundationskills.skills import register_builtin_skills

        register_builtin_skills(REGISTRY)
        mapping: dict[str, str] = {}
        for name in REGISTRY.names():
            try:
                skill = REGISTRY.get(name)
            except Exception:  # noqa: BLE001 - broken registration must not break attribution
                continue
            for spec in getattr(skill, "rules", ()):
                mapping.setdefault(str(spec.rule_id), str(getattr(skill, "name", name)))
        return mapping

    def _declared_map(self) -> dict[str, str]:
        """rule_id -> the skill whose SKILL.md declares it (used for entry-level ``cited_by``)."""
        mapping = dict(self._registered_declared())
        mapping.update({spec.rule_id: self.name for spec in self.rules})
        if self._declared_override is not None:
            for rule_id in self._declared_override:
                mapping.setdefault(rule_id, self.name)
        return mapping

    @property
    def declared_rule_ids(self) -> frozenset[str]:
        """The overt declared-id set: the override when given, else own rules + every registered skill's rules."""
        if self._declared_override is not None:
            return frozenset(self._declared_override)
        return frozenset(self._declared_map())

    def _citable(self) -> frozenset[str]:
        """Ids that may be cited: ``declared_rule_ids`` plus this skill's own rules (RuleSpec semantics)."""
        return frozenset(self.declared_rule_ids) | {spec.rule_id for spec in self.rules}

    # -- inputs ---------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings = request.get("findings")
        if not isinstance(findings, list):
            return [self.finding(
                "RS-IN-001",
                f"missing input: findings is {type(findings).__name__}, not a list of findings (nothing to surface)",
                {"findings_type": type(findings).__name__},
                "pass the RED/REFUSED findings as [{message, rule_id, severity, evidence, recovery}]",
            )]
        if not findings:
            return [self.finding(
                "RS-IN-001",
                "missing input: findings is empty (nothing to surface)",
                {"findings": []},
                "refuse rather than file an empty surface: pass the findings to attribute",
            )]
        for index, item in enumerate(findings):
            message = item.get("message") if isinstance(item, dict) else None
            if not isinstance(message, str) or not message.strip():
                return [self.finding(
                    "RS-IN-001",
                    f"precondition failed: findings[{index}] is not a finding (no message)",
                    {"entry_index": index},
                    "give every finding a non-empty message; RS-IN-001 refuses rather than surface junk",
                )]
        return []

    # -- run ------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        citable = self._citable()
        declaring = self._declared_map()
        entries: list[dict[str, Any]] = []
        invented: list[str] = []
        attributed: list[Finding] = []
        for index, item in enumerate(request.get("findings") or []):
            entry = dict(item) if isinstance(item, dict) else {"message": str(item)}
            raw_id = entry.get("rule_id")
            raw_severity = entry.get("severity")
            severity_known = raw_severity in _SEVERITIES
            severity = Severity(raw_severity).value if severity_known else "BLOCK"
            id_written = isinstance(raw_id, str) and bool(raw_id.strip())
            if id_written and raw_id not in citable:
                cite = "RS-HO-001"
                invented.append(raw_id)
            elif not id_written:
                cite = "RS-HO-002"
                attributed.append(self.finding(
                    "RS-HO-002",
                    f"finding {index} has no rule id; attributed in the emitted set as RS-HO-002 rather than dropped",
                    {"entry_index": index, "source_rule_id": None if not id_written else raw_id,
                     "severity_normalized": not severity_known},
                    "give the finding a declared rule id (see every SKILL.md's Validation rules)",
                ))
            elif not severity_known or not _RULE_ID.fullmatch(raw_id):
                cite = "RS-HO-002"
                attributed.append(self.finding(
                    "RS-HO-002",
                    f"finding {index} is malformed ({'unknown severity' if not severity_known else 'id does not match ^[A-Z]{2,4}-[A-Z0-9_-]+$'}); "
                    "cited in the emitted set as RS-HO-002",
                    {"entry_index": index, "source_rule_id": raw_id, "severity_normalized": not severity_known},
                    "cite the declared id / a BLOCK-WARN-INFO severity (see every SKILL.md's Validation rules)",
                ))
            else:
                cite = raw_id
            evidence = dict(entry.get("evidence")) if isinstance(entry.get("evidence"), dict) else {}
            evidence["source_rule_id"] = None if raw_id is None or raw_id == "" else raw_id
            message = entry.get("message")
            if not isinstance(message, str) or not message.strip():
                message = f"finding without a message (entry {index})"
            entries.append({
                "rule_id": cite,
                "severity": severity,
                "message": message,
                "evidence": evidence,
                "recovery": str(entry.get("recovery") or ""),
                "cited_by": declaring.get(cite, self.name),
            })

        if not entries:  # reached only on a direct run() call: refuse rather than emit an empty surface
            return self._refused(
                "RS-IN-001: missing input: findings is empty (nothing to surface)",
                [self.finding("RS-IN-001", "missing input: findings is empty (nothing to surface)", {"findings": []})],
            )

        source_skill = str(request.get("source_skill") or "")
        result_status = str(request.get("result_status") or "REFUSED")
        payload = {"source_skill": source_skill, "result_status": result_status, "findings": entries}
        seal_hash = sha256_json(entries)
        artifact_id = str(request.get("artifact_id") or "") or f"rs-{seal_hash[:12]}"
        provenance = make_provenance(self.name, self.version,
                                     {"finding_set": seal_hash, "source_skill": source_skill})
        ref = write_artifact(
            Artifact(type="finding_set", id=artifact_id, payload=payload, provenance=provenance),
            self._artifact_dir(request, ctx),
        )

        if invented:  # an invented/undeclared id is RED: the entry still records what arrived
            unique = list(dict.fromkeys(invented))
            finding = self.finding(
                "RS-HO-001",
                "undeclared rule id(s) invented: " + ", ".join(unique) + " -- no registered skill declares them",
                {"invented_rule_ids": unique},
                "cite the declared id or declare it with a MUST_FIRE fixture",
            )
            return SkillResult(
                status=Status.RED,
                payload=payload,
                artifacts=(ref,),
                findings=(finding,),
                next_actions=("cite the declared rule id (or declare it with a MUST_FIRE fixture) and reseal",),
                provenance=provenance,
            )
        return SkillResult(
            status=Status.PASS,
            payload=payload,
            artifacts=(ref,),
            findings=tuple(attributed),
            next_actions=() if not attributed else ("attribute the flagged findings to declared rule ids before filing",),
            provenance=provenance,
        )

    @staticmethod
    def _artifact_dir(request: dict[str, Any], ctx: SkillContext) -> Path:
        """``out`` names the directory (or a ``*.json`` path whose parent) for the sealed set; default <workdir>/artifacts."""
        out = request.get("out")
        if not isinstance(out, str) or not out.strip():
            return ctx.artifacts_dir
        path = Path(out)
        directory = path.parent if path.suffix == ".json" else path
        if str(directory) in (".", ""):
            return ctx.workdir
        return directory if directory.is_absolute() else ctx.workdir / directory

    # -- handoff --------------------------------------------------------------------------------
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        """Belt-and-braces: every emitted entry must cite a rule id a registered skill declares."""
        citable = self._citable()
        entries: list[dict[str, Any]] = []
        payload = result.payload if isinstance(result.payload, dict) else {}
        for item in payload.get("findings") or []:
            if isinstance(item, dict):
                entries.append(item)
        for ref in result.artifacts:
            try:
                sealed = read_artifact(ref.path, expect_type="finding_set")
            except Exception:  # noqa: BLE001 - unreadable here means "no entry to audit", never a fabrication
                continue
            for item in sealed.payload.get("findings") or []:
                if isinstance(item, dict):
                    entries.append(item)
        return [
            self.finding(
                "RS-HO-001",
                f"emitted entry cites undeclared rule id {item.get('rule_id')!r}",
                {"rule_id": item.get("rule_id")},
                "cite the declared id or declare it with a MUST_FIRE fixture",
            )
            for item in entries
            if item.get("rule_id") not in citable
        ]

    # -- diagnosis ------------------------------------------------------------------------------
    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"{type(failure).__name__}: {failure}" if isinstance(failure, BaseException)
                   else f"refusal_surface returned {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(
            symptom=symptom,
            likely_causes=("the model wrote its own rule id",
                           "a finding lost its id in serialisation"),
            checks=("look up the id in every SKILL.md's Validation rules",),
            recovery=("cite the declared id or declare it with a MUST_FIRE fixture",),
        )

    # -- fixtures -------------------------------------------------------------------------------
    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        return {
            "RS-IN-001": {"request": {"findings": []},
                          "expected": "REFUSED 96 naming RS-IN-001"},
            "RS-HO-001": {"request": {"findings": [{"rule_id": "PF-IN-999", "severity": "BLOCK", "message": "invented"}]},
                          "expected": "RED 5 with RS-HO-001; result.findings contains no PF-IN-999; "
                                      "payload entry cites RS-HO-001 carrying source_rule_id: PF-IN-999"},
            "RS-HO-002": {"request": {"findings": [{"message": "no id at all"}]},
                          "expected": "PASS 0 with RS-HO-002; emitted entry rule_id: RS-HO-002"},
        }
