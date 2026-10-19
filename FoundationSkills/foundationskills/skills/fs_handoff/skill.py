"""FsHandoffSkill: the FoundationSkills session handoff writer (A3.2 user-invoked workflow skill, /fs-handoff).

Exit contract: 96 REFUSED writes nothing and names the missing input (FH-IN-001: no `artifacts` list, so there is
nothing to hand off; FH-IN-002: an artifact path is missing or is not an artifact envelope with `type` and `payload`);
0 PASS when every artifact is referenced BY PATH and `notes`/`next_steps` are redacted. The skill writes one `handoff`
artifact, composes and invokes no other skill (``invokes`` is empty) and its handoff-phase rule FH-HO-001 is INFO
(redaction counts travel with the result and never block), so RED 5 never fires here.
"""
from __future__ import annotations

import hashlib
import json
import re
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

# -- redaction patterns ----------------------------------------------------------------------------------------------
# secrets: sk-..., ghp_..., AKIA..., "Bearer <token>", and NAME=value lines whose NAME holds KEY/TOKEN/SECRET/PASSWORD.
# an assignment value runs to the end of its line (or its quotes) and is redacted while the NAME= is kept.
_SECRET: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("env_value", re.compile(
        r"(?im)^([ \t]*[A-Za-z0-9_\-.]*(?:KEY|TOKEN|SECRET|PASSWORD)[A-Za-z0-9_\-.]*[ \t]*=[ \t]*)"
        r'''("[^"]*"|'[^']*'|[^\r\n]+)''')),
    ("sk_token", re.compile(r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_\-]{4,}")),
    ("ghp_token", re.compile(r"(?<![A-Za-z0-9_])ghp_[A-Za-z0-9]{8,}")),
    ("akia_key", re.compile(r"(?<![A-Za-z0-9])AKIA[A-Z0-9]{12,}")),
    ("bearer_token", re.compile(r"(?i)Bearer[ \t]+[A-Za-z0-9._\-+/=]{4,}")),
)
# PII: the data_engine builtin (reused, never re-implemented) with these five types.
_PII: frozenset[str] = frozenset({"email", "phone", "credit_card", "ssn", "national_id"})

_REFERENCE_FIELDS = ("path", "type", "id", "sha256")


class FsHandoffSkill(BaseSkill):
    name = "fs_handoff"
    version = "0.1.0"
    description = (
        "Slash-command closer (/fs-handoff) for a FoundationSkills session: takes this session's artifact JSON "
        "paths plus optional `notes` and `next_steps` and writes ONE handoff artifact that references every "
        "artifact BY PATH ({path, type, id, sha256}, never a copied payload) with `notes`/`next_steps` redacted "
        "of secrets (sk-, ghp_, AKIA, Bearer, NAME=<KEY/TOKEN/SECRET/PASSWORD> values) and PII (email, phone, "
        "credit_card, ssn, national_id). Side-effect free elsewhere: it composes and invokes nothing and writes "
        "only its own artifact. Do NOT use for re-running a skill, fixing findings or deciding what to run next."
    )
    scope = Scope(  # hands any session's artifacts off, so it spans every axis the catalogue does
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "dgx-h100", "local"),
    )
    consumes = ()
    produces = ("handoff",)
    invocation = "user"
    invokes = ()
    requires_fs = False

    input_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "artifacts": {"type": "array", "items": {"type": "string", "minLength": 1}},
            "notes": {"type": "string"},
            "next_steps": {"type": "array", "items": {"type": "string"}},
            "record_dir": {"type": "string", "minLength": 1},
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "required": ["artifacts", "notes", "next_steps", "redactions"],
        "additionalProperties": True,
        "properties": {
            "artifacts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": list(_REFERENCE_FIELDS),
                    "additionalProperties": False,
                    "properties": {
                        "path": {"type": "string"},
                        "type": {"type": "string"},
                        "id": {"type": "string"},
                        "sha256": {"type": "string"},
                    },
                },
            },
            "notes": {"type": "string"},
            "next_steps": {"type": "array", "items": {"type": "string"}},
            "redactions": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}},
        },
    }
    rules = (
        RuleSpec("FH-IN-001", 'no "artifacts" list is given, so there is nothing to hand off', Severity.BLOCK, "input"),
        RuleSpec("FH-IN-002", 'an artifact path is missing or is not an artifact envelope (no type/payload)', Severity.BLOCK, "input"),
        RuleSpec("FH-HO-001", "redactions were applied to notes and next_steps: report the counts; never blocks", Severity.INFO, "handoff"),
    )
    fs_interface = FSInterface(
        entries=("fskills handoff",),
        emits=(),
        apis=(),
        notes=("references session artifacts by path and writes one redacted handoff: it copies no payload, runs no "
               "other skill and touches no cluster resource"),
    )

    # -- helpers -----------------------------------------------------------------------------------------------
    @staticmethod
    def _read_artifact(path: Any) -> tuple[dict[str, Any] | None, str | None]:
        """Read one artifact file into {path, type, id, provenance, sha256}; (None, error) when it is missing or not
        an artifact envelope with `type` and `payload` — that error fires FH-IN-002 and names the path.

        The envelope's type, id and provenance are read and the file's bytes are hashed, but the payload is never
        copied: the handoff references artifacts BY PATH and keeps only the reference fields.
        """
        label = str(path)
        try:
            raw = Path(label).read_bytes()
        except Exception as exc:  # noqa: BLE001 - any read failure is a missing artifact
            return None, f"artifact {label!r} is missing or unreadable: {exc}"
        try:
            envelope = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - bytes that are not JSON cannot be an envelope
            return None, f"artifact {label!r} is not an artifact envelope: its bytes are not JSON ({exc})"
        if not isinstance(envelope, dict) or "type" not in envelope or "payload" not in envelope:
            keys = sorted(str(k) for k in envelope) if isinstance(envelope, dict) else type(envelope).__name__
            return None, (f"artifact {label!r} is not an artifact envelope: it carries no `type` and `payload` "
                          f"(keys: {keys})")
        return {
            "path": label,
            "type": str(envelope.get("type")),
            "id": str(envelope.get("id") or Path(label).stem),
            "provenance": envelope.get("provenance"),  # read (it pins the writer); never copied into the payload
            "sha256": hashlib.sha256(raw).hexdigest(),
        }, None

    @staticmethod
    def _writer(provenance: Any) -> str:
        """The writer recorded in an envelope's provenance (read, summarised, never copied)."""
        if isinstance(provenance, dict):
            for key in ("writer", "skill", "name"):
                if provenance.get(key):
                    return str(provenance[key])
            return "unknown"
        return str(provenance or "unknown")

    @staticmethod
    def _redact_text(text: str) -> tuple[str, dict[str, int]]:
        """`text` with secrets and PII replaced by [REDACTED:<kind>], plus the per-kind counts."""
        from foundationskills.skills.data_engine.ops.clean import _redact_pii_builtin  # noqa: PLC0415

        counts: dict[str, int] = {}
        out = text
        for kind, pattern in _SECRET:  # env-assignments first: NAME= is kept, its value is redacted whole
            if kind == "env_value":
                out, hits = pattern.subn(lambda m: f"{m.group(1)}[REDACTED:env_value]", out)
            else:
                out, hits = pattern.subn(lambda m, k=kind: f"[REDACTED:{k}]", out)
            if hits:
                counts[kind] = counts.get(kind, 0) + int(hits)
        out, pii = _redact_pii_builtin(out, set(_PII))
        for kind, hits in (pii or {}).items():
            if int(hits):
                counts[str(kind)] = counts.get(str(kind), 0) + int(hits)
        return out, counts

    @staticmethod
    def _merge(target: dict[str, int], found: dict[str, int]) -> None:
        for kind, hits in found.items():
            if int(hits):
                target[str(kind)] = target.get(str(kind), 0) + int(hits)

    # -- inputs ------------------------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        paths = request.get("artifacts")
        if not isinstance(paths, list) or not paths:
            findings.append(self.finding(
                "FH-IN-001",
                "refused: missing input: no \"artifacts\" list is given, so there is nothing to hand off ("
                f"{'no `artifacts` in the request' if paths is None else f'{paths!r} is not a non-empty list of artifact paths'})",
                {"artifacts": str(paths)},
                "pass `artifacts` (a list of artifact JSON paths from this session), with `notes` and `next_steps` for the next session",
            ))
            return findings
        for path in paths:
            _, error = self._read_artifact(path)
            if error:
                findings.append(self.finding(
                    "FH-IN-002",
                    f"refused: {error}",
                    {"path": str(path)},
                    "point the list at artifacts written this session (each a JSON envelope with `type` and `payload`), then re-run fs_handoff",
                ))
        return findings

    # -- run ---------------------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        references: list[dict[str, Any]] = []   # BY PATH: {path, type, id, sha256} only — never a payload
        writers: list[str] = []
        for path in request.get("artifacts") or ():
            envelope, _ = self._read_artifact(path)
            if envelope is None:
                continue  # unreachable: check_inputs refused before run
            references.append({key: envelope[key] for key in _REFERENCE_FIELDS})
            writers.append(self._writer(envelope.get("provenance")))

        redacted_notes, counts = self._redact_text(str(request.get("notes") or ""))
        redacted_steps: list[str] = []
        for step in request.get("next_steps") or ():
            redacted, found = self._redact_text(str(step))
            redacted_steps.append(redacted)
            self._merge(counts, found)

        payload: dict[str, Any] = {
            "artifacts": references,
            "notes": redacted_notes,
            "next_steps": redacted_steps,
            "redactions": dict(sorted(counts.items())),
        }
        provenance = make_provenance(
            self.name, self.version,
            {
                "artifact_paths": ",".join(item["path"] for item in references) or "",
                "artifact_types": ",".join(item["type"] for item in references) or "",
                "source_writers": ",".join(sorted(set(writers))) or "",
                "redaction_kinds": ",".join(sorted(counts)) or "none",
            },
        )
        artifact = Artifact("handoff", "handoff", payload, provenance, 1)
        ref = write_artifact(artifact, request.get("record_dir") or ctx.artifacts_dir)
        return SkillResult(
            status=Status.PASS, payload=payload, artifacts=(ref,), findings=(),
            next_actions=(), provenance=provenance,
        )

    # -- handoff -----------------------------------------------------------------------------------------------
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        """Fires FH-HO-001 (INFO, handoff phase) from the PASS payload's redaction counts; it reports and never blocks."""
        counts = {str(k): int(v) for k, v in dict(((result.payload or {}).get("redactions")) or {}).items()}
        applied = {kind: hits for kind, hits in counts.items() if hits}
        if not applied:
            return []
        return [self.finding(
            "FH-HO-001",
            (f"redactions applied to notes and next_steps before the handoff: "
             f"{', '.join(f'{kind} {hits}' for kind, hits in sorted(applied.items()))} "
             f"({sum(applied.values())} total) — secrets and PII never travel in a handoff"),
            {"redactions": ",".join(f"{kind}:{hits}" for kind, hits in sorted(applied.items()))},
            "read the redacted handoff before sharing it, keep secrets and PII out of `notes`/`next_steps`, and pass the referenced artifacts on by path",
        )]

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"fs_handoff {type(failure).__name__}: {failure}"
                   if isinstance(failure, BaseException)
                   else f"fs_handoff {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(
            symptom=symptom,
            likely_causes=(
                "no `artifacts` list was given (or it was empty), so there is nothing to hand off (FH-IN-001)",
                "an artifact path is missing, or the file is not an artifact envelope with `type` and `payload` (FH-IN-002)",
            ),
            checks=(
                "read the FH-IN-001/FH-IN-002 finding message (it names the missing input or the bad path)",
                "check every path is a JSON envelope written this session (type, id, provenance, payload)",
            ),
            recovery=(
                "pass `artifacts` (artifact JSON paths from this session), with `notes` and `next_steps` for the next session",
                "fix or drop the named path so every entry is a real artifact envelope, then re-run fs_handoff",
            ),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        return {
            "FH-IN-001": {
                "request": {"artifacts": [], "notes": "carry on where we stopped", "next_steps": ["resume the trainer at step 3"]},
                "expected": "REFUSED 96 naming FH-IN-001 (no `artifacts` list is given, so there is nothing to hand off); nothing is written",
            },
            "FH-IN-002": {
                "request": {"artifacts": ["job/artifacts/probe.json", "job/scratch/raw.json"],
                            "notes": "carry on"},
                "expected": "REFUSED 96 naming FH-IN-002 (job/scratch/raw.json is not an artifact envelope with type and payload); nothing is written",
            },
            "FH-HO-001": {
                "request": {"artifacts": ["job/artifacts/probe.json"],
                            "notes": "the curator is a.chen@example.com and the key is sk-live-9f2c44ab77",
                            "next_steps": ["resume the trainer at step 3"],
                            "record_dir": "job/records"},
                "simulated_setup": "one artifact envelope at job/artifacts/probe.json",
                "expected": "PASS 0 writing one handoff artifact with FH-HO-001 INFO naming the counts (secret_sk 1, email 1); it reports and never blocks",
            },
        }
