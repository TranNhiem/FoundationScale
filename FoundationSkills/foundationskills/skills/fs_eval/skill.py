"""FsEvalSkill: the /fs-eval model-skill workflow (A3.2 user-invoked workflow skill).

Composes exactly ONE model skill — ``evaluation`` — and only through ``foundationskills.skills._workflow``
(``run_step`` / ``outcome_class`` / ``registry`` / ``Step``): no other skill's ``execute()`` is ever called
here and no inner finding surfaces — its rule id is undeclared in fs_eval and would turn this result RED, so
the Step rows travel as ``evidence["steps"]`` of our own declared rules. Outcome mapping, exactly as the
``_workflow`` docstring states:

    inner REFUSED     -> FE-IN-002 (input BLOCK)   -> REFUSED 96, nothing written (no workflow_record)
    inner RED         -> FE-HO-001 (handoff BLOCK) -> RED 5 (check_handoff reads payload["red_step"])
    inner UNMEASURED  -> FE-HO-002 (WARN)         -> UNMEASURED 95, never PASS
    inner PASS        ->                           -> PASS 0

FE-IN-001 (input BLOCK) fires from the request alone: neither "checkpoint" nor "run_manifest" is given.
Exit contract: 96 REFUSED writes nothing; 0 PASS writes one ``workflow_record`` whose ``outputs`` name the
evaluation's ``eval_report`` BY PATH (never copied); 5 RED (FE-HO-001) and 95 UNMEASURED (FE-HO-002) write
the same record (``verdict`` "RED" / "UNMEASURED") and are never usable evaluation output.
Composition rule: a user skill composes model skills, never another user skill — fs_eval composes
``evaluation`` and nothing else (``invokes = ("evaluation",)``).
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
from foundationskills.skills._workflow import Step, outcome_class, run_step

_VERDICT = {"ok": "PASS", "red": "RED", "unmeasured": "UNMEASURED", "refused": "REFUSED"}
_PATH_KEYS = ("path", "uri", "location", "filename")
_ID_KEYS = ("name", "type", "kind", "id", "artifact_type")


class FsEvalSkill(BaseSkill):
    name = "fs_eval"
    version = "0.1.0"
    description = (
        "/fs-eval is the A3.2 user-invoked workflow skill: it takes the evaluation request (checkpoint, "
        "run_manifest, base, benchmarks, policy, eval_cache, out, baseline_cache, ...) and forwards it "
        "UNCHANGED to the ONE composed model skill `evaluation`, run through foundationskills.skills._workflow, "
        "then writes one workflow_record {workflow, steps, outputs, verdict} whose outputs name the "
        "eval_report BY PATH. Outcome mapping (PASS 0 / RED 5 / UNMEASURED 95 / REFUSED 96): inner REFUSED "
        "-> FE-IN-002 REFUSED; inner RED -> FE-HO-001 RED; inner UNMEASURED -> FE-HO-002 UNMEASURED (never "
        "PASS). Do NOT use for fixing a regression, running a bare lm-eval harness or judging "
        "launch-readiness: /fs-eval records the evaluation verdict, it never repairs a benchmark or tunes a "
        "policy."
    )
    scope = Scope(  # identical to fs_status's: it records any evaluation result the catalogue can produce
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "dgx-h100", "local"),
    )
    consumes = ("checkpoint",)
    produces = ("workflow_record",)
    invocation = "user"
    invokes = ("evaluation",)
    requires_fs = True

    input_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "checkpoint": {"type": "string", "minLength": 1},
            "base": {"type": "string", "minLength": 1},
            "run_manifest": {"type": "string", "minLength": 1},
            "benchmarks": {"type": "array", "items": {"type": "string", "minLength": 1}},
            "policy": {"type": "string", "minLength": 1},
            "eval_cache": {"type": "string", "minLength": 1},
            "out": {"type": "string", "minLength": 1},
            "baseline_cache": {"type": "string", "minLength": 1},
            "num_fewshot": {"type": "integer", "minimum": 0},
            "seed": {"type": "integer"},
            "gen_kwargs": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1},
            "dtype": {"enum": ["bfloat16", "float16", "float32"]},
            "batch_size": {"type": "string"},
            "device": {"type": "string", "minLength": 1},
            "parallelize": {"type": "boolean"},
            "include_path": {"type": "string"},
            "heldout": {"type": "object"},
            "data": {"type": "string", "minLength": 1},
            "max_length": {"type": "integer", "minimum": 1},
            "video_frames": {"type": ["integer", "null"], "minimum": 1},
            "image_column": {"type": "string", "minLength": 1},
            "video_column": {"type": "string", "minLength": 1},
            "heldout_script": {"type": ["string", "null"], "minLength": 1},
            "abs_epsilon": {"type": "number", "minimum": 0},
            "python": {"type": "string", "minLength": 1},
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "required": ["workflow", "steps", "outputs", "verdict"],
        "additionalProperties": True,
        "properties": {
            "workflow": {"type": "string", "minLength": 1},
            "steps": {"type": "array"},
            "outputs": {
                "type": "object",
                "properties": {"eval_report": {"type": ["string", "null"]}},
            },
            "verdict": {"type": "string", "minLength": 1},
        },
    }
    rules = (
        RuleSpec("FE-IN-001", 'neither "checkpoint" nor "run_manifest" is given', Severity.BLOCK, "input"),
        RuleSpec("FE-IN-002", "the composed evaluation REFUSED", Severity.BLOCK, "input"),
        RuleSpec("FE-HO-001", "the composed evaluation is RED (a regression beyond the policy band, a broken report)", Severity.BLOCK, "handoff"),
        RuleSpec("FE-HO-002", "the composed evaluation is UNMEASURED", Severity.WARN, "handoff"),
    )
    fs_interface = FSInterface(
        entries=("fskills eval",),
        emits=(),
        apis=(),
        notes=("composes the evaluation model skill (which drives FoundationScale's harness) through "
               "foundationskills.skills._workflow; writes only its own workflow_record and references the "
               "eval_report by path"),
    )

    # -- helpers ---------------------------------------------------------------------------------
    @staticmethod
    def _report_path(step: Step) -> str | None:
        """Path of the composed evaluation's eval_report — referenced BY PATH, never copied."""
        refs = list(step.artifacts)
        for ref in refs:
            ident = " ".join(str(ref.get(key) or "") for key in _ID_KEYS)
            if "eval_report" in ident:
                for key in _PATH_KEYS:
                    if ref.get(key):
                        return str(ref[key])
        if len(refs) == 1:  # a lone artifact is the evaluation's report regardless of how it is labelled
            for key in _PATH_KEYS:
                if refs[0].get(key):
                    return str(refs[0][key])
        return None

    def _ev_in_001(self, request: dict[str, Any]) -> Finding:
        return self.finding(
            "FE-IN-001",
            'missing input: neither "checkpoint" nor "run_manifest" is given, so the composed evaluation '
            "has no work to run",
            {"checkpoint": str(request.get("checkpoint") or ""), "run_manifest": str(request.get("run_manifest") or "")},
            'pass `checkpoint` (a checkpoint directory) or `run_manifest` (a run manifest naming one) together '
            "with the evaluation fields it needs (`eval_cache`, `out`, ...), then re-run /fs-eval",
        )

    # -- inputs ---------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        has_checkpoint = bool(str(request.get("checkpoint") or "").strip())
        has_manifest = bool(str(request.get("run_manifest") or "").strip())
        if not (has_checkpoint or has_manifest):
            return [self._ev_in_001(request)]
        return []

    # -- run ------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        inner, step = run_step(self, "evaluation", request, ctx)  # the ONE composed model skill
        cls = outcome_class(inner)  # "ok" | "refused" | "red" | "unmeasured" -> our own declared rule
        rows = [step.as_row()]
        outputs = {"eval_report": self._report_path(step)}
        payload: dict[str, Any] = {"workflow": "/fs-eval", "steps": rows, "outputs": outputs,
                                   "verdict": _VERDICT[cls]}
        provenance = make_provenance(
            self.name, self.version,
            {
                "checkpoint": str(request.get("checkpoint") or ""),
                "run_manifest": str(request.get("run_manifest") or ""),
                "step": step.skill,
                "step_status": step.status,
            },
        )
        # inner REFUSED -> FE-IN-002 (input BLOCK) -> REFUSED 96 through the refusal path: nothing is written
        if cls == "refused":
            finding = self.finding(
                "FE-IN-002",
                f"refused: the composed evaluation REFUSED — {step.refusal or 'it named no reason'}",
                {"steps": rows, "refusal": step.refusal, "outputs": outputs},
                "give the evaluation exactly what its refusal names (a checkpoint, a resolvable base model, "
                "`eval_cache`, `out`, ...), then re-run /fs-eval",
            )
            return SkillResult(
                status=Status.REFUSED, payload={}, artifacts=(), findings=(finding,), refusal=finding.message,
                next_actions=(), provenance=provenance,
            )
        if cls == "red":
            payload["red_step"] = True  # the flag check_handoff reads to fire FE-HO-001
        ref = write_artifact(
            Artifact("workflow_record", "workflow", payload, provenance, 1),
            # `out` is passed through to evaluation as ITS report file; measured 2026-10-18 on GPU: writing this
            # record into it crashed /fs-eval after the evaluation itself passed. The record has its own key.
            request.get("record_dir") or ctx.artifacts_dir,
        )
        # inner RED -> FE-HO-001 (handoff BLOCK) -> RED 5, fired in check_handoff from payload["red_step"]
        if cls == "red":
            return SkillResult(
                status=Status.PASS, payload=payload, artifacts=(ref,), findings=(),  # check_handoff turns red_step RED
                next_actions=(), provenance=provenance,
            )
        # inner UNMEASURED -> FE-HO-002 (WARN) -> UNMEASURED 95, never PASS
        if cls == "unmeasured":
            finding = self.finding(
                "FE-HO-002",
                "the composed evaluation is UNMEASURED: a benchmark or the held-out hit was not measured, "
                "so this is not a pass",
                {"steps": rows, "outputs": outputs},
                "resolve what the evaluation's findings name (a skipped benchmark, a limited/smoke run) and "
                "re-run /fs-eval — never treat UNMEASURED as PASS",
            )
            return SkillResult(
                status=Status.UNMEASURED, payload=payload, artifacts=(ref,), findings=(finding,),
                next_actions=(), provenance=provenance,
            )
        return SkillResult(
            status=Status.PASS, payload=payload, artifacts=(ref,), findings=(),
            next_actions=(), provenance=provenance,
        )

    # -- handoff --------------------------------------------------------------------------------
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        rows = list(result.payload.get("steps") or ())
        if result.payload.get("red_step"):  # FE-HO-001: inner RED -> handoff BLOCK -> RED 5
            findings.append(self.finding(
                "FE-HO-001",
                "handoff BLOCK: the composed evaluation is RED (a benchmark breached its policy band or the "
                "eval_report is broken), so its output must not be used",
                {"steps": rows, "outputs": dict(result.payload.get("outputs") or {})},
                "fix what the evaluation's findings name and re-run that evaluation through /fs-eval — never "
                "treat RED as usable output",
            ))
        return findings

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"fs_eval {type(failure).__name__}: {failure}"
                   if isinstance(failure, BaseException)
                   else f"fs_eval {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(
            symptom=symptom,
            likely_causes=(
                'neither "checkpoint" nor "run_manifest" was given (FE-IN-001)',
                "the composed evaluation REFUSED: its refusal names the missing input (FE-IN-002)",
                "the composed evaluation is RED (FE-HO-001): a benchmark breached its policy band or the report is broken",
                "the composed evaluation is UNMEASURED (FE-HO-002): a benchmark or the held-out hit was not measured",
            ),
            checks=(
                'read the step rows in the workflow_record (or in the FE-* finding\'s evidence["steps"]): each '
                "names the skill, its status/exit_code and its refusal",
                "check the request carries an eval_cache, an out directory and a resolvable base model — the "
                "evaluation's EV-IN-001..EV-IN-008 name what is missing",
            ),
            recovery=(
                "fix what the refusal or the findings name, then re-run /fs-eval",
                "hand the artifacts (eval_report BY PATH) on only when the verdict is PASS — never treat "
                "UNMEASURED or RED as PASS",
            ),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        return {
            "FE-IN-001": {
                "request": {},
                "expected": 'REFUSED 96 naming FE-IN-001 (neither "checkpoint" nor "run_manifest" is given) — '
                            "fires from the request alone, nothing is written",
            },
            "FE-IN-002": {
                "request": {"checkpoint": "/models/q3", "eval_cache": "/eval/cache", "out": "/out/q3"},
                "simulated_step": {"skill": "evaluation", "status": "REFUSED", "exit_code": 96,
                                  "refusal": "EV-IN-003: eval cache /eval/cache is missing"},
                "expected": "REFUSED 96 naming FE-IN-002 (the composed evaluation REFUSED) with the step row in "
                            "evidence['steps'] — no workflow_record is written",
            },
            "FE-HO-001": {
                "request": {"checkpoint": "/models/q3", "eval_cache": "/eval/cache", "out": "/out/q3"},
                "simulated_step": {"skill": "evaluation", "status": "RED", "exit_code": 5, "rule": "EV-HO-001"},
                "expected": "RED 5 naming FE-HO-001 (the composed evaluation is RED) after check_handoff reads "
                            'payload["red_step"]; the workflow_record verdict is "RED"',
            },
            "FE-HO-002": {
                "request": {"checkpoint": "/models/q3", "eval_cache": "/eval/cache", "out": "/out/q3"},
                "simulated_step": {"skill": "evaluation", "status": "UNMEASURED", "exit_code": 95,
                                  "rule": "EV-HO-003"},
                "expected": "UNMEASURED 95 naming FE-HO-002 (the composed evaluation is UNMEASURED) — the WARN "
                            'finding travels with the result and the verdict is never "PASS"',
            },
        }
