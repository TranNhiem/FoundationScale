"""FsStatusSkill: the FoundationSkills status explainer (A3.2 user-invoked workflow skill, /fs-status).

Exit contract: 96 REFUSED writes nothing and names the missing input (ST-IN-001: no readable journal
record and no "result" object; ST-IN-002: the record's status and exit_code disagree with the doctrine in
core/status.py, so it cannot be explained honestly); 0 PASS when the explanation is measured against the
doctrine. The skill writes one status_explanation artifact, invokes no other skill and declares no handoff
rule, so RED 5 never fires here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from foundationskills.core.artifacts import Artifact, write_artifact
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
from foundationskills.core.provenance import make_provenance
from foundationskills.core.status import Status

_MEANING: dict[str, str] = {
    "PASS": "the work ran and every check passed",
    "RED": "the work ran but a check failed, so the output must not be used",
    "UNMEASURED": "something that matters was not measured, so it is not a pass",
    "REFUSED": "nothing ran; an input or precondition was missing",
}

_NEXT_STEP: dict[str, str] = {
    "PASS": "hand the artifacts on to the next skill in the chain",
    "RED": "fix what the findings name, then re-run that skill — never treat RED as usable output",
    "UNMEASURED": "resolve what the findings name, then re-run that skill — never treat UNMEASURED as PASS",
    "REFUSED": "add the input or precondition the refusal names, then re-run that skill",
}

_KNOWN_EXIT = {status.value: status.exit_code for status in Status}


class FsStatusSkill(BaseSkill):
    name = "fs_status"
    version = "0.1.0"
    description = (
        "Slash-command explainer (/fs-status) for FoundationSkills results: takes a journal.jsonl record "
        "written by the orchestrator (last record by default, or a `step_index`) or a skill result object "
        "{status, exit_code, refusal, findings} and returns one plain-English `status_explanation` saying "
        "what the status means (PASS 0, RED 5, UNMEASURED 95, REFUSED 96) and what to do next. Side-effect "
        "free: it composes and invokes nothing, writes only its own artifact. Do NOT use for fixing the "
        "problem, re-running skills, or deciding whether to launch: it explains, it does not act."
    )
    scope = Scope(  # explains any status, so it spans every axis the catalogue does
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "dgx-h100", "local"),
    )
    consumes = ()
    produces = ("status_explanation",)
    invocation = "user"
    invokes = ()
    requires_fs = False

    input_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "journal": {"type": "string", "minLength": 1},
            "step_index": {"type": "integer", "minimum": 0},
            "result": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "minLength": 1},
                    "exit_code": {"type": "integer"},
                    "refusal": {"type": ["string", "null"]},
                    "findings": {"type": "array"},
                    "skill": {"type": "string", "minLength": 1},
                },
            },
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "required": ["source", "status", "exit_code", "meaning", "next_step"],
        "additionalProperties": True,
        "properties": {
            "source": {"type": "string"},
            "skill": {"type": ["string", "null"]},
            "status": {"type": "string"},
            "exit_code": {"type": "integer"},
            "meaning": {"type": "string"},
            "next_step": {"type": "string"},
            "refusal": {"type": ["string", "null"]},
            "findings": {"type": "array"},
            "artifact_ids": {"type": "array"},
        },
    }
    rules = (
        RuleSpec("ST-IN-001", "neither a readable journal record nor a \"result\" object is given", Severity.BLOCK, "input"),
        RuleSpec("ST-IN-002", "the record's status and exit_code disagree with the doctrine", Severity.BLOCK, "input"),
    )
    fs_interface = FSInterface(
        entries=("fskills status",),
        emits=(),
        apis=(),
        notes="side-effect free explainer: reads a journal record or a result object and writes one status_explanation",
    )

    # -- helpers ---------------------------------------------------------------------------------
    @staticmethod
    def _journal_record(request: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        """Load one record from a journal.jsonl (last record, or `step_index`).

        Returns (record, error): record is None exactly when the journal cannot be read
        (missing file, bad JSON line, step_index out of range) — that error triggers ST-IN-001.
        """
        path = request.get("journal")
        if not path:
            return None, None
        try:
            lines = [line for line in Path(str(path)).read_text().splitlines() if line.strip()]
            records = [json.loads(line) for line in lines]
        except Exception as exc:  # noqa: BLE001 - any read/parse failure is an unreadable journal
            return None, f"journal {str(path)!r} is not readable: {exc}"
        if not records:
            return None, f"journal {str(path)!r} holds no records"
        index = request.get("step_index")
        if index is None:
            index = len(records) - 1
        if not 0 <= int(index) < len(records):
            return None, f"step_index {index} is out of range for journal {str(path)!r} ({len(records)} records)"
        return records[int(index)], None

    @staticmethod
    def _payload_record(record: dict[str, Any]) -> dict[str, Any]:
        """Journal record -> the explanation payload shape (doctrine-checked in check_inputs)."""
        status = str(record.get("status") or "")
        return {
            "source": "journal",
            "skill": str(record.get("skill")) if record.get("skill") else None,
            "status": status,
            "exit_code": int(record.get("exit_code", _KNOWN_EXIT.get(status, -1))),
            "meaning": _MEANING.get(status, ""),
            "next_step": _NEXT_STEP.get(status, ""),
            "refusal": str(record.get("refusal")) if record.get("refusal") else None,
            "findings": [],
            "artifact_ids": [str(a) for a in (record.get("artifact_ids") or ())],
        }

    @staticmethod
    def _payload_result(result: dict[str, Any]) -> dict[str, Any]:
        """Result object -> the explanation payload shape (doctrine-checked in check_inputs)."""
        status = str(result.get("status") or "")
        findings = []
        for raw in result.get("findings") or ():
            raw = raw if isinstance(raw, dict) else {}
            findings.append({
                "rule_id": str(raw.get("rule_id") or ""),
                "severity": str(raw.get("severity") or ""),
                "message": str(raw.get("message") or ""),
                "recovery": str(raw.get("recovery")) if raw.get("recovery") else None,
            })
        return {
            "source": "result",
            "skill": str(result.get("skill")) if result.get("skill") else None,
            "status": status,
            "exit_code": int(result.get("exit_code", _KNOWN_EXIT.get(status, -1))),
            "meaning": _MEANING.get(status, ""),
            "next_step": _NEXT_STEP.get(status, ""),
            "refusal": str(result.get("refusal")) if result.get("refusal") else None,
            "findings": findings,
            "artifact_ids": [],
        }

    @staticmethod
    def _doctrine_error(status: Any, exit_code: Any) -> str | None:
        """None when `status`/`exit_code` agree with core/status.py; else the disagreement (ST-IN-002)."""
        known = _KNOWN_EXIT.get(str(status or ""))
        if known is None:
            return f"unknown status {status!r} (the only statuses are PASS/RED/UNMEASURED/REFUSED)"
        if exit_code is not None and int(exit_code) != known:
            return (f"status {status!r} maps to exit_code {known} in the doctrine, "
                    f"but the record says {exit_code!r}: an inconsistent record cannot be explained honestly")
        return None

    # -- inputs ---------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        record, error = self._journal_record(request)
        result = request.get("result") if isinstance(request.get("result"), dict) else None
        if record is None and result is None:
            findings.append(self.finding(
                "ST-IN-001",
                f"missing input: neither a readable journal record nor a result object is given ({error or 'no `journal` and no `result` in the request'})",
                {"journal": str(request.get("journal") or ""), "has_result": bool(result)},
                "pass `journal` (a journal.jsonl path, with optional `step_index`) or a `result` object {status, exit_code, refusal, findings}",
            ))
            return findings
        payload = self._payload_record(record) if record is not None else self._payload_result(result)
        disagreement = self._doctrine_error(payload["status"], payload["exit_code"])
        if disagreement:
            findings.append(self.finding(
                "ST-IN-002",
                f"refused: the record's status and exit_code disagree with the doctrine — {disagreement}",
                {"status": payload["status"], "exit_code": payload["exit_code"]},
                "fix the record so its status and exit_code agree with the doctrine (PASS 0, RED 5, UNMEASURED 95, REFUSED 96)",
            ))
        return findings

    # -- run ------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        record, _ = self._journal_record(request)
        result = request.get("result") if isinstance(request.get("result"), dict) else None
        payload = self._payload_record(record) if record is not None else self._payload_result(result)
        provenance = make_provenance(
            self.name, self.version,
            {
                "journal": str(request.get("journal") or ""),
                "step_index": str(request.get("step_index") or ""),
                "result_status": str((result or {}).get("status") or ""),
            },
        )
        artifact = Artifact("status_explanation", "status", payload, provenance, 1)
        ref = write_artifact(artifact, request.get("out") or ctx.artifacts_dir)
        return SkillResult(
            status=Status.PASS, payload=payload, artifacts=(ref,), findings=(),
            next_actions=(), provenance=provenance,
        )

    # -- handoff --------------------------------------------------------------------------------
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        return []  # the status_explanation payload is schema-validated on write; no handoff rule is declared

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"fs_status {type(failure).__name__}: {failure}"
                   if isinstance(failure, BaseException)
                   else f"fs_status {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(
            symptom=symptom,
            likely_causes=(
                "neither a readable journal record nor a result object was given (missing file, bad JSON line, step_index out of range)",
                "the record's status and exit_code disagree with the doctrine (PASS 0, RED 5, UNMEASURED 95, REFUSED 96)",
            ),
            checks=(
                "read the ST-IN-001/ST-IN-002 finding message (it names the missing or inconsistent input)",
                "check the journal path holds JSONL records with status, exit_code and artifact_ids",
            ),
            recovery=(
                "pass `journal` (with an in-range `step_index`) or a `result` object",
                "fix the record so its status matches its exit code, then re-run fs_status",
            ),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        return {
            "ST-IN-001": {
                "request": {"journal": "/nonexistent/journal.jsonl"},
                "expected": "REFUSED 96 naming ST-IN-001 (neither a readable journal record nor a result object is given)",
            },
            "ST-IN-002": {
                "request": {"result": {"status": "PASS", "exit_code": 5}},
                "expected": "REFUSED 96 naming ST-IN-002 (the record's status and exit_code disagree with the doctrine)",
            },
        }
