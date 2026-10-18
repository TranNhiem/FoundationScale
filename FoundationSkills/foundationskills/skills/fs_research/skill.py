"""FsResearchSkill: the claim-measurement research workflow (A3.2 user-invoked workflow skill, /fs-research).

Exit contract: 96 REFUSED writes nothing and names what is missing or refused — FR-IN-001: no "claims" list, or
an empty one; FR-IN-002: the composed measured_or_unmeasured step refused (its findings travel inside
evidence["steps"], never as this skill's findings). 95 UNMEASURED (FR-HO-001 WARN, returned from run()) when some
corrected claim is still UNMEASURED — a claim nobody measured is never a pass. 5 RED only from check_handoff
(FR-HO-002 BLOCK on a PASS result whose payload flag "composed_red" the composed step's RED set): run() never
returns Status.RED, it hands the flag to check_handoff. 0 PASS when the composed step passed and every corrected
claim is measured. Outcome precedence: REFUSED (FR-IN-001 / FR-IN-002) > RED (FR-HO-002) > UNMEASURED (FR-HO-001)
> PASS. The artifact is written to request["record_dir"] or ctx.artifacts_dir, never to a composed skill's key.
"""
from __future__ import annotations

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
from foundationskills.skills._workflow import artifact_paths, load_payload, outcome_class, run_step

_CLAIM_ITEM: dict[str, Any] = {  # exactly as measured_or_unmeasured takes them
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
}


def _claim_status(claim: Any) -> str:
    claim = claim if isinstance(claim, dict) else {}
    for key in ("asserted_status", "status"):
        value = claim.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return ""


def _is_unmeasured(claim: Any) -> bool:
    """A corrected claim is unmeasured when nothing measured what it asserts."""
    claim = claim if isinstance(claim, dict) else {}
    return claim.get("measured") is False or _claim_status(claim) == "UNMEASURED"


def _corrected_claims(composed: SkillResult, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The composed step's corrected claims: its claim_set artifact payload, read through _workflow.load_payload."""
    paths = artifact_paths(composed, "claim_set")
    if not paths:
        return list(fallback)  # the composed step wrote no claim_set: audit the claims as given
    try:
        payload = load_payload(paths[0])
    except Exception:  # noqa: BLE001 - an unreadable artifact must not crash the research
        return list(fallback)
    if isinstance(payload, dict) and isinstance(payload.get("claims"), list):
        return list(payload["claims"])
    return list(payload) if isinstance(payload, list) else list(fallback)


class FsResearchSkill(BaseSkill):
    name = "fs_research"
    version = "0.1.0"
    description = (
        "Claim-measurement research workflow (/fs-research): takes an optional goal_spec path and a claims list "
        "{id, claim, asserted_status?, evidence?, measured?}, composes measured_or_unmeasured to check every claim, "
        "and writes one research_findings artifact {goal_spec, claims (corrected), verdict (PASS|UNMEASURED), steps}. "
        "REFUSED (96) with no or empty claims (FR-IN-001) or when the composed step refuses (FR-IN-002); "
        "UNMEASURED (95) when some claim stays UNMEASURED (FR-HO-001); RED (5) when the composed step came back RED "
        "(FR-HO-002). Do NOT use for training or launching (fskills-training) or re-running the checker alone "
        "(fskills-measured-or-unmeasured): /fs-research researches and reports, the trainer launches."
    )
    scope = Scope(  # any claim set for any run, so it spans every axis the catalogue does
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "dgx-h100", "local"),
    )
    consumes = ("goal_spec", "claim_set")
    produces = ("research_findings",)
    invocation = "user"
    invokes = ("measured_or_unmeasured",)
    requires_fs = False

    input_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "goal_spec": {"type": "string", "minLength": 1},
            "claims": {"type": "array", "items": dict(_CLAIM_ITEM)},
            "record_dir": {"type": "string", "minLength": 1},
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "required": ["goal_spec", "claims", "verdict", "steps"],
        "additionalProperties": True,
        "properties": {
            "goal_spec": {"type": ["string", "null"]},
            "claims": {"type": "array", "items": dict(_CLAIM_ITEM)},
            "verdict": {"enum": ["PASS", "UNMEASURED"]},
            "steps": {"type": "array"},
            "composed_status": {"enum": ["PASS", "RED", "UNMEASURED", "REFUSED"]},
            "composed_red": {"type": "boolean"},
        },
    }
    rules = (
        RuleSpec("FR-IN-001", 'no "claims" list is given, or the list is empty', Severity.BLOCK, "input"),
        RuleSpec("FR-IN-002", "the composed measured_or_unmeasured step refused", Severity.BLOCK, "input"),
        RuleSpec("FR-HO-001", "some corrected claims are UNMEASURED", Severity.WARN, "handoff"),
        RuleSpec("FR-HO-002", "the composed measured_or_unmeasured step came back RED", Severity.BLOCK, "handoff"),
    )
    fs_interface = FSInterface(
        entries=("fskills research",),
        emits=(),
        apis=(),
        notes=(
            "composes measured_or_unmeasured through foundationskills.skills._workflow.run_step (never another user "
            "skill), reads its claim_set artifact and writes one research_findings artifact; imports no FoundationScale "
            "and touches no cluster resource"
        ),
    )

    # -- helpers ---------------------------------------------------------------------------------
    @staticmethod
    def _payload(goal_spec: str | None, claims: list[dict[str, Any]], verdict: str,
                 steps: list[dict[str, Any]], composed_status: str) -> dict[str, Any]:
        return {
            "goal_spec": goal_spec,
            "claims": [c if isinstance(c, dict) else {} for c in claims],
            "verdict": "UNMEASURED" if verdict != "PASS" else "PASS",
            "steps": list(steps),
            "composed_status": str(composed_status),
            "composed_red": str(composed_status).upper() == "RED",  # the payload flag check_handoff reads
        }

    # -- inputs ---------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        claims = request.get("claims")
        if not isinstance(claims, list) or not claims:
            count = len(claims) if isinstance(claims, list) else 0
            return [self.finding(
                "FR-IN-001",
                f'refused: no "claims" list is given, or the list is empty ({count} claims)',
                {"has_claims": isinstance(claims, list), "claim_count": count,
                 "goal_spec": str(request.get("goal_spec") or "")},
                'pass `claims` as a non-empty list of claim objects: [{"id": "c1", "claim": "the loss fell", '
                '"measured": true}], then re-run /fs-research',
            )]
        return []

    # -- run ------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        claims = [c if isinstance(c, dict) else {} for c in (request.get("claims") or ())]
        goal_spec = str(request.get("goal_spec")) if request.get("goal_spec") else None
        output_root = request.get("record_dir") or ctx.artifacts_dir
        composed, step = run_step(self, "measured_or_unmeasured", {"claims": claims}, ctx)
        steps = [step.as_row()]
        outcome = outcome_class(composed)
        inputs = {"goal_spec": goal_spec or "", "claims": len(claims), "composed_status": composed.status.value}

        if outcome == "refused":  # inner REFUSED -> the input-phase BLOCK rule -> REFUSED (96), nothing written
            reason = "; ".join(str(f.get("message") or "") for f in step.findings if f.get("message"))
            reason = reason or str(composed.refusal) or "the composed step gave no reason"
            finding = self.finding(
                "FR-IN-002",
                f"refused: the composed measured_or_unmeasured step was refused — {reason}",
                {"steps": steps},
                'give measured_or_unmeasured the input its refusal names (its findings are in evidence["steps"]), '
                "then re-run /fs-research",
            )
            return SkillResult(
                status=Status.REFUSED,
                payload=self._payload(goal_spec, [], "UNMEASURED", steps, composed.status.value),
                artifacts=(),
                findings=(finding,),
                next_actions=(),
                provenance=make_provenance(self.name, self.version, inputs),
                refusal=finding.message,
            )

        corrected = _corrected_claims(composed, claims)
        unmeasured = [c for c in corrected if _is_unmeasured(c)]
        verdict = "UNMEASURED" if (unmeasured or outcome == "unmeasured") else "PASS"
        payload = self._payload(goal_spec, corrected, verdict, steps, composed.status.value)
        provenance = make_provenance(self.name, self.version, {**inputs, "verdict": verdict})
        written = (write_artifact(
            Artifact("research_findings", "research_findings", payload, provenance, 1), output_root,
        ),)

        if outcome == "red":  # inner RED -> the handoff-phase BLOCK rule -> RED (5): flag only, never RED from run()
            return SkillResult(
                status=Status.PASS, payload=payload, artifacts=written, findings=(), next_actions=(),
                provenance=provenance,
            )

        if verdict == "UNMEASURED":  # WARN: a finding from run() with UNMEASURED, never PASS
            named = ", ".join(str(c.get("id")) for c in unmeasured if isinstance(c, dict))
            why: list[str] = []
            if unmeasured:
                why.append(f"{len(unmeasured)} of {len(corrected)} corrected claims are UNMEASURED ({named})")
            if outcome == "unmeasured":
                why.append("the composed measured_or_unmeasured step itself came back UNMEASURED")
            finding = self.finding(
                "FR-HO-001",
                "unmeasured: " + "; ".join(why),
                {"unmeasured_ids": [str(c.get("id")) for c in unmeasured if isinstance(c, dict)],
                 "composed_status": composed.status.value},
                "measure what the named claims assert (see claims[].evidence) and re-run the claims checker — "
                "never treat UNMEASURED as PASS",
            )
            return SkillResult(
                status=Status.UNMEASURED, payload=payload, artifacts=written, findings=(finding,), next_actions=(),
                provenance=provenance,  # refusal is for REFUSED only (SkillResult contract)
            )

        return SkillResult(
            status=Status.PASS, payload=payload, artifacts=written, findings=(), next_actions=(),
            provenance=provenance,
        )

    # -- handoff --------------------------------------------------------------------------------
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        """FR-HO-002 fires ONLY here, on the PASS result whose payload flag composed_red is set -> RED 5."""
        payload = result.payload if isinstance(result.payload, dict) else {}
        if not payload.get("composed_red"):
            return []
        return [self.finding(
            "FR-HO-002",
            'red: the composed measured_or_unmeasured step came back RED, so the research_findings must not be '
            'used (its findings are in evidence["steps"])',
            {"steps": payload.get("steps") or [], "composed_status": str(payload.get("composed_status") or "")},
            'read the findings the RED names (evidence["steps"]), fix them with that skill and re-run /fs-research — '
            "never hand RED research_findings on",
        )]

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"fs_research {type(failure).__name__}: {failure}"
                   if isinstance(failure, BaseException)
                   else f"fs_research {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(
            symptom=symptom,
            likely_causes=(
                'no "claims" list was given, or the list was empty (missing key, or [] was passed)',
                "the composed measured_or_unmeasured step was refused (its findings name the input it needs)",
                "some corrected claims are still UNMEASURED (nothing measured what those claims assert)",
                "the composed measured_or_unmeasured step came back RED (a check failed inside the claims checker)",
            ),
            checks=(
                "read the FR-IN-001/FR-IN-002/FR-HO-001/FR-HO-002 finding message (it names the claim ids and steps)",
                "read research_findings.steps to see what the composed measured_or_unmeasured step wrote",
            ),
            recovery=(
                'pass `claims` as a non-empty list of {"id", "claim", ...} objects and re-run /fs-research',
                'give measured_or_unmeasured the input its refusal names (evidence["steps"]) and re-run /fs-research',
                "measure the UNMEASURED claims and re-run the claims checker — never treat UNMEASURED as PASS",
                "fix what the composed findings name and re-run /fs-research — never hand RED research_findings on",
            ),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        return {
            "FR-IN-001": {
                "request": {"goal_spec": "goals/measure_loss.goal.json", "record_dir": "out/research"},
                "expected": 'REFUSED 96 naming FR-IN-001 (no "claims" list is given, or the list is empty); nothing is written',
            },
            "FR-IN-002": {
                "request": {"record_dir": "out/research",
                            "claims": [{"id": "c1", "claim": "the loss fell", "measured": True}]},
                "simulated_step": "measured_or_unmeasured refused its input",
                "expected": 'REFUSED 96 naming FR-IN-002 (the composed measured_or_unmeasured step was refused); its '
                            'findings travel in evidence["steps"] and nothing is written',
            },
            "FR-HO-001": {
                "request": {"goal_spec": "goals/rl.goal.json", "record_dir": "out/research",
                            "claims": [{"id": "c2", "claim": "the reward rose", "measured": False}]},
                "simulated_step": "claims checker found unmeasured claims",
                "expected": 'UNMEASURED 95 naming FR-HO-001 (some corrected claims are UNMEASURED) with one '
                            'research_findings artifact whose verdict is "UNMEASURED"',
            },
            "FR-HO-002": {
                "request": {"goal_spec": "goals/q2.goal.json", "record_dir": "out/research",
                            "claims": [{"id": "c1", "claim": "the loss fell", "measured": True}]},
                "simulated_step": "measured_or_unmeasured went RED",
                "expected": "RED 5 naming FR-HO-002 from check_handoff on the PASS run whose payload flag "
                            "composed_red is set (run() never returns RED)",
            },
        }
