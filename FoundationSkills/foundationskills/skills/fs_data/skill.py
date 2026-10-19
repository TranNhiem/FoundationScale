"""FsDataSkill: the FoundationSkills data preparation workflow composer (A3.2 user-invoked workflow skill, /fs-data).

Exit contract: 96 REFUSED writes nothing and names the missing input (FD-IN-001: no "sources" and no explicit
"pipeline"; FD-IN-002: the composed data_engine refused); 5 RED when data_engine came back RED (FD-HO-001: zero
records or readiness verdict RED); 95 UNMEASURED when data_engine came back UNMEASURED (FD-HO-002: readiness checks
not run); 0 PASS when data_engine returned PASS. The skill writes one workflow_record artifact (verdict
PASS/RED/UNMEASURED; nothing is written for REFUSED) and composes exactly one model skill: data_engine.
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
from foundationskills.skills._workflow import Step, outcome_class, registry, run_step


class FsDataSkill(BaseSkill):
    name = "fs_data"
    version = "0.1.0"
    description = (
        "Slash-command data preparation workflow (/fs-data) that composes the model skill 'data_engine' to "
        "turn raw data refs into a dataset, data_pipeline_spec and readiness_report. Inputs: the data_engine "
        "request fields (sources, target_format, tokenizer, chat_template_family, out_dir, requirements, "
        "pipeline, reasoning_unsupported, benchmarks, ...) passed through unchanged, plus optional 'goal_spec' "
        " (a path recorded in outputs, not interpreted). The workflow record says which step ran, what it "
        "produced and the verdict (PASS/RED/UNMEASURED). Do NOT use for explaining fskills results (use "
        "/fs-status), fixing data problems yourself, or training a model (use /fs-training): this composes "
        "data_engine and produces data artifacts only."
    )
    scope = Scope(  # identical to fs_status: any model type, family and stage the catalogue covers
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "dgx-h100", "local"),
    )
    consumes = ("raw_data_ref",)
    produces = ("workflow_record",)
    invocation = "user"
    invokes = ("data_engine",)
    requires_fs = False

    input_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "sources": {"type": "array"},
            "target_format": {},
            "tokenizer": {},
            "chat_template_family": {},
            "out_dir": {},
            "requirements": {"type": "object"},
            "pipeline": {"type": "object"},
            "reasoning_unsupported": {},
            "benchmarks": {},
            "goal_spec": {"type": "string", "minLength": 1},
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "required": ["workflow", "steps", "outputs", "verdict"],
        "additionalProperties": True,
        "properties": {
            "workflow": {"type": "string"},
            "steps": {"type": "array"},
            "outputs": {
                "type": "object",
                "properties": {
                    "dataset": {"type": ["string", "null"]},
                    "readiness_report": {"type": ["string", "null"]},
                    "data_pipeline_spec": {"type": ["string", "null"]},
                    "goal_spec": {"type": ["string", "null"]},
                },
            },
            "verdict": {"type": "string", "enum": ["PASS", "RED", "UNMEASURED"]},
        },
    }
    rules = (
        RuleSpec("FD-IN-001", 'no "sources" and no explicit "pipeline" is given', Severity.BLOCK, "input"),
        RuleSpec("FD-IN-002", "the composed data_engine refused", Severity.BLOCK, "input"),
        RuleSpec("FD-HO-001", "data_engine came back RED (e.g. zero records, readiness RED)", Severity.BLOCK, "handoff"),
        RuleSpec("FD-HO-002", "data_engine came back UNMEASURED (readiness checks not run)", Severity.WARN, "handoff"),
    )
    fs_interface = FSInterface(
        entries=("fskills data",),
        emits=(),
        apis=(),
        notes="composes data_engine to turn raw data refs into a dataset with a readiness report",
    )

    # -- helpers ---------------------------------------------------------------------------------
    @staticmethod
    def _outputs_from_artifacts(result: SkillResult, request: dict[str, Any]) -> dict[str, Any]:
        """Map the inner data_engine artifacts to named output paths (by path, never copied) plus goal_spec."""
        outputs: dict[str, Any] = {"dataset": None, "readiness_report": None, "data_pipeline_spec": None}
        for ref in result.artifacts:
            info: dict[str, Any] = ref.to_dict() if hasattr(ref, "to_dict") else dict(ref)  # type: ignore[arg-type]
            kind = str(info.get("kind") or info.get("type") or info.get("name") or "")
            path = str(info.get("path") or info.get("uri") or info.get("location") or "")
            if kind == "dataset":
                outputs["dataset"] = path or None
            elif kind == "readiness_report":
                outputs["readiness_report"] = path or None
            elif kind == "data_pipeline_spec":
                outputs["data_pipeline_spec"] = path or None
        if request.get("goal_spec"):
            outputs["goal_spec"] = str(request["goal_spec"])
        return outputs

    # -- inputs -----------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        if not request.get("sources") and not request.get("pipeline"):
            findings.append(self.finding(
                "FD-IN-001",
                'missing input: no "sources" and no explicit "pipeline" is given — the data_engine '
                "needs at least one data source or an explicit pipeline definition to run",
                {"sources": list(request.get("sources") or ()), "has_pipeline": bool(request.get("pipeline"))},
                'pass "sources" (a list of data source URIs) or an explicit "pipeline" object',
            ))
        return findings

    # -- run ---------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        result, step = run_step(self, "data_engine", request, ctx, registry())
        cls = outcome_class(result)
        steps: list[dict[str, Any]] = [step.as_row()]
        outputs = self._outputs_from_artifacts(result, request)
        provenance = make_provenance(
            self.name, self.version,
            {
                "sources_count": str(len(request.get("sources") or ())),
                "target_format": str(request.get("target_format") or ""),
                "goal_spec": str(request.get("goal_spec") or ""),
            },
        )

        # ---- inner REFUSED -> FD-IN-001 INPUT BLOCK -> REFUSED (nothing written) ------------------
        if cls == "refused":
            finding = self.finding(
                "FD-IN-002",
                f"the composed data_engine refused — {result.refusal or 'see its findings'}",
                {"steps": steps},
                "fix what the data_engine findings or refusal name and re-run",
            )
            return SkillResult(
                status=Status.REFUSED, payload={}, artifacts=(), findings=(finding,), refusal=finding.message,
                next_actions=(), provenance=provenance,
            )

        verdict_map: dict[str, str] = {"ok": "PASS", "red": "RED", "unmeasured": "UNMEASURED"}
        verdict = verdict_map[cls]

        payload: dict[str, Any] = {
            "workflow": "fs_data",
            "steps": steps,
            "outputs": outputs,
            "verdict": verdict,
        }
        # payload["red_step"] is the flag that check_handoff reads to fire FD-HO-001
        if cls == "red":
            payload["red_step"] = True

        artifact = Artifact("workflow_record", "workflow_record", payload, provenance, 1)
        ref = write_artifact(artifact, request.get("record_dir") or ctx.artifacts_dir)  # never a passthrough key

        # ---- inner UNMEASURED -> FD-HO-002 WARN -> UNMEASURED (never PASS) -----------------------
        if cls == "unmeasured":
            finding = self.finding(
                "FD-HO-002",
                "data_engine came back UNMEASURED — readiness checks were not run, so the dataset is "
                "not a confirmed pass and must not be treated as PASS",
                {"steps": steps},
                "resolve the readiness issues and re-run; never treat UNMEASURED as PASS",
            )
            return SkillResult(
                status=Status.UNMEASURED, payload=payload, artifacts=(ref,), findings=(finding,),
                next_actions=(), provenance=provenance,
            )

        # ---- inner PASS -> PASS ------------------------------------------------------------------
        return SkillResult(
            status=Status.PASS, payload=payload, artifacts=(ref,), findings=(),
            next_actions=(), provenance=provenance,
        )

    # -- handoff -----------------------------------------------------------------------------------
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        if result.payload.get("red_step"):
            steps = result.payload.get("steps", [])
            findings.append(self.finding(
                "FD-HO-001",
                "data_engine came back RED — e.g. zero records were written or the readiness verdict is RED",
                {"steps": steps},
                "fix the data_engine findings and re-run; never treat a RED dataset as ready for training",
            ))
        return findings

    # -- diagnostics -------------------------------------------------------------------------------
    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"fs_data {type(failure).__name__}: {failure}"
                   if isinstance(failure, BaseException)
                   else f"fs_data {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(
            symptom=symptom,
            likely_causes=(
                'no "sources" and no explicit "pipeline" was given (FD-IN-001)',
                "the composed data_engine refused (FD-IN-002) — e.g. unknown target_format, missing local source",
                "data_engine returned RED (FD-HO-001) — e.g. zero records, readiness RED",
                "data_engine returned UNMEASURED (FD-HO-002) — readiness checks were not run",
            ),
            checks=(
                "read the finding message (it names FD-IN-001/FD-IN-002/FD-HO-001/FD-HO-002 and shows the step evidence)",
                "check the workflow_record artifact for the per-step status and outputs",
                "check the data_engine rules (DE-IN-001..DE-IN-008, DE-HO-001..DE-HO-003) for the root cause",
            ),
            recovery=(
                'pass "sources" or an explicit "pipeline" (FD-IN-001)',
                "fix the data_engine input issues it named (e.g. unknown target_format DE-IN-002, missing source DE-IN-003) (FD-IN-002)",
                "resolve the data_engine findings and re-run (FD-HO-001 / FD-HO-002)",
                "never treat UNMEASURED or RED as a ready dataset — fix and re-run",
            ),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        return {
            "FD-IN-001": {
                "request": {},
                "expected": 'REFUSED 96 naming FD-IN-001 (no "sources" and no explicit "pipeline" is given)',
            },
            "FD-IN-002": {
                "request": {"sources": [{"uri": "/data/raw/missing_corpus.jsonl", "kind": "jsonl"}], "target_format": "sft"},
                "expected": "REFUSED 96 naming FD-IN-002 (the composed data_engine refused)",
            },
            "FD-HO-001": {
                "request": {"sources": [{"uri": "/data/raw/chat_corpus.jsonl", "kind": "jsonl"}], "target_format": "sft"},
                "simulated_step": "data_engine wrote zero records",
                "expected": "RED 5 naming FD-HO-001 (data_engine came back RED)",
            },
            "FD-HO-002": {
                "request": {"sources": [{"uri": "/data/raw/chat_corpus.jsonl", "kind": "jsonl"}], "target_format": "sft"},
                "simulated_step": "readiness checks were not run",
                "expected": "UNMEASURED 95 with FD-HO-002 WARN finding (data_engine came back UNMEASURED)",
            },
        }
