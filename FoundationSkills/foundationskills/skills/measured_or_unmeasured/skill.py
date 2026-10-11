"""MeasuredOrUnmeasuredSkill: rewrite a claim set so nothing unevidenced reads as PASS.

Exit contract: exit set 0/95 only — 0 PASS when every claim names an evidence
pointer that resolves to a measured JSON field on disk, 95 UNMEASURED as soon as
one claim is demoted (MU-IN-001/002/003 warn and travel with the result;
absence of evidence is never PASS). The corrected claim set is sealed as
``claim_set.<artifact_id>.json`` with provenance. This skill never upgrades an
unmeasured claim and never invents evidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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
from foundationskills.core.provenance import make_provenance, sha256_json
from foundationskills.core.status import Status, worst
from foundationskills.skills.measured_or_unmeasured.evidence import EvidenceError, resolve_evidence


class MeasuredOrUnmeasuredSkill(BaseSkill):
    name = "measured_or_unmeasured"
    version = "0.1.0"
    description = (
        "Rewrite a claim set so every claim without a measured evidence pointer "
        "(<artifact path>#<json field> resolvable on disk) is emitted UNMEASURED, never a silent PASS."
    )
    scope = Scope(
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=(),
        hardware=("local",),
    )
    consumes = ("claim_set",)
    produces = ("claim_set",)

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["claims"],
        "additionalProperties": False,
        "properties": {
            "claims": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["id", "claim"],
                    "additionalProperties": True,
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "claim": {"type": "string", "minLength": 1},
                        "asserted_status": {"enum": ["PASS", "RED", "UNMEASURED", "REFUSED"]},
                        "evidence": {"type": ["string", "null"]},
                        "measured": {"type": "boolean"},
                    },
                },
            },
            "out": {"type": "string", "minLength": 1},
            "artifact_id": {"type": "string", "minLength": 1},
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "required": ["verdict", "claims"],
        "additionalProperties": True,
        "properties": {
            "verdict": {"enum": ["PASS", "UNMEASURED"]},
            "claims": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["id", "claim", "status"],
                    "additionalProperties": True,
                    "properties": {
                        "status": {"enum": ["MEASURED", "UNMEASURED"]},
                        "evidence": {"type": ["string", "null"]},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    }
    rules = (
        RuleSpec("MU-IN-001", "a claim asserts a measured PASS (or measured: true) with no evidence pointer (emitted UNMEASURED)", Severity.WARN, "input"),
        RuleSpec("MU-IN-002", "an evidence pointer is malformed or names no measured value on disk (claim demoted)", Severity.WARN, "input"),
        RuleSpec("MU-IN-003", "a claim carries no verdict at all (absence of evidence emitted as UNMEASURED)", Severity.WARN, "input"),
    )
    fs_interface = FSInterface(
        entries=(),
        emits=("claim_set",),
        apis=(),
        notes="pure rewrite; no FoundationScale call",
    )

    def __init__(self, resolve: Callable[[str], Any] = resolve_evidence) -> None:
        self._resolve = resolve

    # -- inputs -------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        for claim in request["claims"]:
            _, claim_findings = self._rewrite(claim)
            findings.extend(claim_findings)
        return findings

    # -- rewrite ------------------------------------------------------------------------
    def _rewrite(self, claim: dict[str, Any]) -> tuple[dict[str, Any], list[Finding]]:
        """Rewrite one claim to MEASURED|UNMEASURED and return the findings its evidence earns."""
        claim_id = claim["id"]
        pointer = claim.get("evidence")
        asserted = claim.get("asserted_status")
        measured_flag = claim.get("measured")

        asserts_pass = asserted == "PASS" or measured_flag is True
        has_verdict = asserted is not None or isinstance(measured_flag, bool)

        findings: list[Finding] = []
        measured = False
        failure = ""
        if pointer is None:
            if asserts_pass:
                findings.append(self.finding(
                    "MU-IN-001",
                    f"claim {claim_id!r} asserts a measured PASS with no evidence pointer; emitted UNMEASURED",
                    {"claim_id": claim_id, "asserted_status": asserted, "measured": measured_flag},
                    "measure it (record <artifact path>#<json field>) and re-run",
                ))
        else:
            try:
                self._resolve(pointer)
            except (EvidenceError, OSError, ValueError) as exc:  # EvidenceError is a ValueError; OSError names an absent file
                failure = str(exc)
                findings.append(self.finding(
                    "MU-IN-002",
                    f"claim {claim_id!r} evidence pointer is malformed or unresolvable: {failure}; emitted UNMEASURED",
                    {"claim_id": claim_id, "evidence": pointer, "error": failure},
                    "point evidence at a measured value on disk as <artifact path>#<json field>, then re-run",
                ))
            else:
                measured = True
        if not has_verdict and not measured:  # a resolved pointer IS the verdict: the claim is MEASURED
            findings.append(self.finding(
                "MU-IN-003",
                f"claim {claim_id!r} carries no verdict at all; emitted UNMEASURED",
                {"claim_id": claim_id, "evidence": pointer},
                "assert a status backed by a measured evidence pointer, then re-run",
            ))

        if measured:
            reason = f"MEASURED: evidence pointer {pointer!r} resolved to a measured value on disk"
        else:
            causes: list[str] = []
            if pointer is None and asserts_pass:
                causes.append("claimed PASS with no evidence pointer (MU-IN-001)")
            if pointer is not None:
                causes.append(f"evidence pointer {pointer!r} names no measured value ({failure}) (MU-IN-002)")
            if not has_verdict:
                causes.append("claim carries no verdict (MU-IN-003)")
            if not causes:
                causes.append("no measured evidence pointer")
            reason = "UNMEASURED: " + "; ".join(causes)

        row = dict(claim)
        row["status"] = "MEASURED" if measured else "UNMEASURED"
        row["evidence"] = pointer
        row["reason"] = reason
        return row, findings

    # -- run ----------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        rows: list[dict[str, Any]] = []
        for claim in request["claims"]:
            row, _ = self._rewrite(claim)  # findings come from check_inputs and travel with the result
            rows.append(row)

        overall = worst(*[Status.PASS if row["status"] == "MEASURED" else Status.UNMEASURED for row in rows])
        payload = {"verdict": "PASS" if overall is Status.PASS else "UNMEASURED", "claims": rows}

        claims_sha = sha256_json(request["claims"])
        artifact_id = request.get("artifact_id") or f"mu-{claims_sha[:12]}"
        artifact = Artifact(
            type="claim_set",
            id=artifact_id,
            payload=payload,
            provenance=make_provenance(self.name, self.version, {"claims": claims_sha}),
        )
        directory = Path(request["out"]) if request.get("out") else ctx.artifacts_dir
        directory.mkdir(parents=True, exist_ok=True)  # the seal target may not exist yet (fresh workdir or out=)
        ref = write_artifact(artifact, directory)
        return SkillResult(
            status=overall,
            payload=payload,
            artifacts=(ref,),
            provenance=make_provenance(self.name, self.version, {"claims": claims_sha, "claim_set": ref.sha256}),
            next_actions=() if overall is Status.PASS
            else ("measure the UNMEASURED claims (record <artifact path>#<json field> on disk) and re-run",),
        )

    # -- handoff ------------------------------------------------------------------------
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        """Guard: a claim still marked MEASURED without a resolved pointer must not exist.

        Impossible by construction — ``_rewrite`` emits MEASURED only when the
        evidence pointer resolved to a measured value on disk — and the guard must
        stay BLOCK-free (this skill's exit set is 0/95), so no findings are ever
        returned here.
        """
        return []

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"{type(failure).__name__}: {failure}" if isinstance(failure, BaseException)
                   else f"measured_or_unmeasured returned {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(
            symptom=symptom,
            likely_causes=("claim was asserted without evidence", "pointer names a file that is not on disk"),
            checks=("read the pointer's `#field`",),
            recovery=("measure it, then re-run",),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        return {
            "MU-IN-001": {"request": {"claims": [{"id": "c1", "claim": "sft holds general ability",
                                                  "asserted_status": "PASS"}]},
                          "expected": "UNMEASURED 95 with MU-IN-001; claim c1 written UNMEASURED, never PASS"},
            "MU-IN-002": {"request": {"claims": [{"id": "c1", "claim": "sft holds general ability",
                                                  "asserted_status": "PASS", "evidence": "just-trust-me"}]},
                          "expected": "UNMEASURED 95 with MU-IN-002 (malformed/unresolvable evidence pointer)"},
            "MU-IN-003": {"request": {"claims": [{"id": "c1", "claim": "sft holds general ability"}]},
                          "expected": "UNMEASURED 95 with MU-IN-003 (no verdict, no evidence)"},
        }
