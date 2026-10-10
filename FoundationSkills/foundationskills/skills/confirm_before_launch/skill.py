"""ConfirmBeforeLaunchSkill: fs_launch_spec + the plan hash the user read back -> confirmation_record.json.

Exit contract: 96 REFUSED writes no record (every refusal names the missing input and never
echoes the expected hash: this gate never auto-fills a confirmation); 0 PASS only when the
user-supplied plan hash matches the plan hash recomputed from the spec payload -- the same
token ``fskills launch --confirm`` requires -- and the spec did not drift between that
recompute and the record seal (CB-HO-001).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

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
from foundationskills.core.orchestrator import plan_hash
from foundationskills.core.provenance import make_provenance, sha256_file
from foundationskills.core.schema import SchemaError
from foundationskills.core.status import Status

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")
_REPEAT = "run `fskills hash --spec <launch_spec>` and repeat the confirmation"
_SEALED = "the spec changed before the confirmation record was sealed"


class SpecError(Exception):
    """``launch_spec`` cannot be read as an fs_launch_spec payload (CB-IN-003: nothing to hash)."""


def load_launch_spec(path: Path) -> dict[str, Any]:
    """Return the fs_launch_spec payload to hash.

    A typed artifact envelope is unwrapped and validated with
    ``read_artifact(path, expect_type="fs_launch_spec")``; anything else is taken as the bare
    payload, unwrapped the same way ``cli._load_payload`` unwraps one. Raises ``SpecError``
    (CB-IN-003: nothing to hash) when the file is missing/unreadable or is not an
    fs_launch_spec.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError) as exc:  # ValueError covers json.JSONDecodeError
        raise SpecError(f"{path} is missing or unreadable ({type(exc).__name__})") from exc
    if not isinstance(raw, dict):
        raise SpecError(f"{path} is not a JSON object, so it is not an fs_launch_spec")
    if "payload" in raw and "type" in raw:
        if not isinstance(raw["payload"], dict):
            raise SpecError(f"{path} has no payload object, so it is not an fs_launch_spec")
        if str(raw["type"]) != "fs_launch_spec":
            raise SpecError(f"{path} is {raw['type']!r}, not an fs_launch_spec")
        try:
            return dict(read_artifact(path, expect_type="fs_launch_spec").payload)
        except (SchemaError, KeyError, OSError, TypeError, ValueError) as exc:
            raise SpecError(f"{path} is not a valid fs_launch_spec ({type(exc).__name__})") from exc
    return dict(raw)  # bare payload: hashed exactly as `fskills hash --spec` hashes it


class ConfirmBeforeLaunchSkill(BaseSkill):
    name = "confirm_before_launch"
    version = "0.1.0"
    description = (
        "Gate an fs_launch_spec behind the user-confirmed plan hash: recomputes the plan hash "
        "(`fskills hash --spec`, the same token `fskills launch --confirm` requires), REFUSES (96) when the "
        "user-supplied hash is absent or mismatches, and seals a confirmation_record with confirmed_by=user "
        "and auto_filled=false when they match. Never submits, never auto-fills, never echoes the expected hash."
    )
    scope = Scope(
        model_types=("llm",),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "dgx-h100", "local"),
    )
    consumes = ("fs_launch_spec",)
    produces = ("confirmation_record",)

    input_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["launch_spec"],
        "properties": {
            "launch_spec": {"type": "string"},     # path of the fs_launch_spec JSON (envelope or bare payload)
            "user_plan_hash": {"type": "string"},  # no minLength: an empty string MUST reach check_inputs (CB-IN-001)
            "confirmation_id": {"type": "string"}, # optional correlation id, recorded in provenance
            "out": {"type": "string"},             # directory for the record (a `.json` value uses its parent)
            "artifact_id": {"type": "string"},     # record id (default cb-<spec_sha256[:12]>)
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "required": ["launch_spec", "confirmed_hash", "auto_filled"],
        "properties": {
            "launch_spec": {"type": "string"},
            "confirmed_hash": {"type": "string"},  # verbatim user input, never a recomputed value
            "auto_filled": {"const": False},
        },
    }
    rules = (
        RuleSpec("CB-IN-001", "user_plan_hash absent or empty (launch without re-showing the hash; never auto-filled)", Severity.BLOCK, "input"),
        RuleSpec("CB-IN-002", "the supplied user_plan_hash is not the plan hash recomputed from the spec", Severity.BLOCK, "input"),
        RuleSpec("CB-IN-003", "the fs_launch_spec is missing, unreadable or not an fs_launch_spec (nothing to hash)", Severity.BLOCK, "input"),
        RuleSpec("CB-HO-001", "the spec drifted on disk between the input recompute and the record seal (the second plan_hash differs)", Severity.BLOCK, "handoff"),
    )
    fs_interface = FSInterface(
        entries=(),
        emits=("confirmation_record",),
        apis=("fskills hash --spec / fskills launch --confirm",),
        notes="recomputes the plan hash (fskills hash --spec, the token fskills launch --confirm requires); never submits, never auto-fills",
    )

    def __init__(self, plan_hash: Callable[[dict[str, Any]], str] = plan_hash) -> None:
        self._plan_hash = plan_hash

    # -- helpers -------------------------------------------------------------------------------
    @staticmethod
    def _spec_path(raw: str, ctx: SkillContext) -> Path:
        """A relative launch_spec resolves against the workdir unless it exists as given (cwd)."""
        path = Path(raw)
        if path.is_absolute():
            return path
        candidate = ctx.workdir / path
        return candidate if candidate.exists() and not path.exists() else path

    @staticmethod
    def _out_dir(request: dict[str, Any], ctx: SkillContext) -> Path:
        """``out`` names the directory the record is written into (a ``.json`` value names its parent)."""
        out = request.get("out")
        if not out:
            return ctx.artifacts_dir
        path = Path(str(out))
        return path.parent if path.suffix == ".json" else path

    def _cb_in_003(self, raw: str, exc: SpecError) -> Finding:
        return self.finding("CB-IN-003", f"missing input: {exc}; {_REPEAT}", {"launch_spec": raw}, _REPEAT)

    def _cb_ho_001_finding(self, raw: str) -> Finding:
        return self.finding(
            "CB-HO-001",
            f"{_SEALED}: the plan hash recomputed at seal time differs; {_REPEAT}",
            {"launch_spec": raw},
            _REPEAT,
        )

    # -- inputs ---------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        raw = str(request["launch_spec"])
        path = self._spec_path(raw, ctx)
        payload: dict[str, Any] | None = None
        try:
            payload = load_launch_spec(path)
        except SpecError as exc:
            findings.append(self._cb_in_003(raw, exc))
        user_hash = str(request.get("user_plan_hash") or "")
        if not user_hash.strip():  # launch without re-showing the hash: the skill never auto-fills it
            findings.append(self.finding(
                "CB-IN-001",
                "missing input: user_plan_hash is absent or empty; launching without re-showing the hash is "
                f"refused and this skill never auto-fills it; {_REPEAT}",
                {"launch_spec": raw},
                _REPEAT,
            ))
        elif payload is not None:
            # The input recompute (the comparison lives here so a mismatch is template-REFUSED, 96).
            # Its value is used only here: it is never placed in evidence, messages or a refusal.
            if user_hash != self._plan_hash(payload):
                findings.append(self.finding(
                    "CB-IN-002",
                    "precondition failed: the supplied user_plan_hash does not match the plan hash "
                    f"recomputed from the spec; {_REPEAT}",
                    {"launch_spec": raw},
                    _REPEAT,
                ))
        return findings

    # -- run ------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        raw = str(request["launch_spec"])
        path = self._spec_path(raw, ctx)
        user_hash = str(request.get("user_plan_hash") or "")
        try:
            payload = load_launch_spec(path)
        except SpecError as exc:  # the spec vanished between the input recompute and the seal
            return self._refused(f"CB-IN-003: {exc}", [self._cb_in_003(raw, exc)])
        # Seal-time recompute: the confirmed token is re-checked here and is never reproduced. A
        # second plan_hash that differs is drift between the input recompute and the record seal.
        try:
            if self._plan_hash(payload) != user_hash:
                return self._refused(f"CB-HO-001: {_SEALED}", [self._cb_ho_001_finding(raw)])
            spec_sha = sha256_file(path)  # a separate drift seal: sha256_file(spec), 64 hex
        except OSError:
            return self._refused(f"CB-HO-001: {_SEALED}", [self._cb_ho_001_finding(raw)])
        artifact_id = str(request.get("artifact_id") or f"cb-{spec_sha[:12]}")
        if not _SAFE_ID.fullmatch(artifact_id):
            return self._refused(
                f"invalid request: artifact_id {artifact_id!r} must match [A-Za-z0-9._-]+ and not start with '.'"
            )
        record = {
            "launch_spec": raw,
            "spec_sha256": spec_sha,
            "confirmed_hash": user_hash,  # verbatim user input, never a recomputed value
            "confirmed_by": "user",
            "auto_filled": False,
        }
        inputs = {"launch_spec": raw, "spec_sha256": spec_sha}
        if request.get("confirmation_id"):
            inputs["confirmation_id"] = str(request["confirmation_id"])
        provenance = make_provenance(self.name, self.version, inputs)
        ref = write_artifact(
            Artifact(type="confirmation_record", id=artifact_id, payload=record, provenance=provenance),
            self._out_dir(request, ctx),
        )
        return SkillResult(
            status=Status.PASS,
            payload=record,
            artifacts=(ref,),
            provenance=provenance,
            next_actions=(f"submit only with `fskills launch --spec {raw} --confirm <the hash the user read back>`",),
        )

    # -- handoff --------------------------------------------------------------------------------
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        """Re-verify the sealed drift seal: sha256_file(spec) == record spec_sha256."""
        record = result.payload
        raw = str(record.get("launch_spec", ""))
        path = self._spec_path(raw, ctx)
        problems: list[str] = []
        try:
            current = sha256_file(path)
        except OSError as exc:
            problems.append(f"{_SEALED}: {raw} cannot be re-verified ({type(exc).__name__})")
        else:
            if current != record.get("spec_sha256"):
                problems.append(f"{_SEALED}: sha256_file differs from the sealed spec_sha256")
        return [self.finding("CB-HO-001", p, {"launch_spec": raw}, _REPEAT) for p in problems]

    # -- diagnosis ------------------------------------------------------------------------------
    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (
            f"{type(failure).__name__}: {failure}"
            if isinstance(failure, BaseException)
            else f"confirm_before_launch returned {failure.status.value}: {failure.refusal or 'see findings'}"
        )
        return Diagnosis(
            symptom=symptom,
            likely_causes=("no user-supplied hash", "the spec was re-emitted after the hash was shown"),
            checks=("`fskills hash --spec <spec>`",),
            recovery=("obtain the hash from the user again",),
        )

    # -- must-fire fixtures ---------------------------------------------------------------------
    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        return {
            "CB-IN-001": {
                "request": {"launch_spec": "launch/fs_launch_spec.q27.json"},
                "expected": "REFUSED 96 naming CB-IN-001; no confirmation_record written and the plan hash never echoed",
            },
            "CB-IN-002": {
                "request": {"launch_spec": "launch/fs_launch_spec.q27.json", "user_plan_hash": "0" * 16},
                "expected": "REFUSED 96 naming CB-IN-002",
            },
            "CB-IN-003": {
                "request": {"launch_spec": "missing.json", "user_plan_hash": "f" * 16},
                "expected": "REFUSED 96 naming CB-IN-003",
            },
            "CB-HO-001": {
                "request": {"launch_spec": "launch/fs_launch_spec.q27.json", "user_plan_hash": "<the hash the user read back>"},
                "tamper": "the spec was rewritten on disk between the input recompute and the record seal",
                "expected": "REFUSED 96 naming CB-HO-001, no confirmation_record written",
            },
        }
