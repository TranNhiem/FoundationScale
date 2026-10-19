"""/fs-plan (A3.2): the user-invoked planning workflow. It composes three model skills in order -
training.planner -> probe_first -> training.emit - and stops at the first step that is not PASS. On PASS it shows
the user the plan hash of every emitted fs_launch_spec, to read and type back to /fs-launch; it never launches and
never fills a hash into a launch.

Outcome mapping (foundationskills.skills._workflow): a composed step REFUSED -> FP-IN-003 (REFUSED); RED, or an
emitted spec with executable false -> FP-HO-001 (RED); UNMEASURED -> FP-HO-002 (UNMEASURED, never PASS).
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
from foundationskills.core.orchestrator import plan_hash
from foundationskills.core.provenance import make_provenance
from foundationskills.core.status import Status
from foundationskills.skills._workflow import artifact_paths, load_payload, outcome_class, registry, run_step

_BACK = "then re-run /fs-plan"


class FsPlanSkill(BaseSkill):
    name = "fs_plan"
    version = "0.1.0"
    description = (
        "User-invoked planning workflow (/fs-plan): composes training.planner, probe_first and training.emit, "
        "stops at the first non-PASS step and, on PASS, shows the plan hash of every emitted fs_launch_spec for the "
        "user to type back to /fs-launch. It never launches."
    )
    invocation = "user"
    invokes = ("training.planner", "probe_first", "training.emit")
    requires_fs = True
    scope = Scope(
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "dgx-h100", "local"),
    )
    consumes = ("goal_spec", "readiness_report", "dataset", "probe_report")
    produces = ("workflow_record",)
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "goal": {"type": "string"},
            "readiness": {"type": "string"},
            "dataset": {"type": "string"},
            "probe_report": {"type": "string"},
            "model": {"type": "string"},
            "output_root": {"type": "string"},
            "hardware_id": {"type": "string"},
            "nodes": {"type": "integer", "minimum": 1},
            "gpus_per_node": {"type": "integer", "minimum": 1},
            "out": {"type": "string"},
        },
        "additionalProperties": False,
    }
    output_schema: dict[str, Any] = {"type": "object", "additionalProperties": True}
    rules = (
        RuleSpec("FP-IN-001", "no goal is given", Severity.BLOCK, "input"),
        RuleSpec("FP-IN-002", "no probe_report is given (probe first: plan only against what FS measurably runs)",
                 Severity.BLOCK, "input"),
        RuleSpec("FP-IN-003", "a composed step (planner, probe_first or emit) refused", Severity.BLOCK, "input"),
        RuleSpec("FP-HO-001", "a composed step came back RED, or an emitted spec is not executable", Severity.BLOCK,
                 "handoff"),
        RuleSpec("FP-HO-002", "a composed step came back UNMEASURED", Severity.WARN, "handoff"),
    )
    fs_interface = FSInterface(entries=("fskills plan", "fskills emit", "fskills hash"), emits=(), apis=(),
                               notes="composes training.planner -> probe_first -> training.emit; shows plan hashes")

    # -- inputs -----------------------------------------------------------------------------------
    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        if not request.get("goal"):
            findings.append(self.finding(
                "FP-IN-001", "missing input: no goal (a goal_spec path) is given; nothing can be planned",
                {"goal": request.get("goal")}, f"pass `goal` (a goal_spec artifact path), {_BACK}"))
        if not request.get("probe_report"):
            findings.append(self.finding(
                "FP-IN-002", "missing input: no probe_report is given; plan only against what the installed "
                "FoundationScale measurably runs", {"probe_report": request.get("probe_report")},
                f"run `fskills probe --deep --json > probe.json` and pass it as `probe_report`, {_BACK}"))
        return findings

    # -- run ---------------------------------------------------------------------------------------
    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        reg = registry()
        steps: list[dict[str, Any]] = []
        outputs: dict[str, Any] = {"training_plan": None, "probe_report": str(request["probe_report"]),
                                   "launch_specs": []}
        provenance = make_provenance(self.name, self.version, {
            "goal": str(request["goal"]), "probe_report": str(request["probe_report"]),
            "dataset": str(request.get("dataset") or ""), "model": str(request.get("model") or "")})

        planner_request: dict[str, Any] = {"goal": load_payload(str(request["goal"])),
                                           "hardware_id": str(request.get("hardware_id") or "gb200")}
        if request.get("readiness"):
            planner_request["readiness"] = load_payload(str(request["readiness"]))
        result, step = run_step(self, "training.planner", planner_request, ctx, reg)
        steps.append(step.as_row())
        stop = self._stop(result, steps, outputs, request, ctx, provenance)
        if stop is not None:
            return stop
        plan_paths = artifact_paths(result, "training_plan")
        outputs["training_plan"] = plan_paths[0] if plan_paths else None

        result, step = run_step(self, "probe_first", {"plan_draft": outputs["training_plan"],
                                                     "probe_report": str(request["probe_report"])}, ctx, reg)
        steps.append(step.as_row())
        stop = self._stop(result, steps, outputs, request, ctx, provenance)
        if stop is not None:
            return stop

        emit_request: dict[str, Any] = {
            "plan": load_payload(outputs["training_plan"]),
            "model": str(request.get("model") or ""),
            "output_root": str(request.get("output_root") or ctx.workdir / "runs"),
            "nodes": int(request.get("nodes") or 1),
            "gpus_per_node": int(request.get("gpus_per_node") or 1),
            "hardware_id": str(request.get("hardware_id") or "gb200"),
            "run_prefix": "fskills",
        }
        if request.get("dataset"):
            emit_request["dataset"] = load_payload(str(request["dataset"]))
        result, step = run_step(self, "training.emit", emit_request, ctx, reg)
        steps.append(step.as_row())
        stop = self._stop(result, steps, outputs, request, ctx, provenance)
        if stop is not None:
            return stop

        not_executable = []
        for path in artifact_paths(result, "fs_launch_spec"):
            spec = load_payload(path)
            row = {"path": path, "stage": spec.get("stage") or spec.get("run_name"), "plan_hash": plan_hash(spec),
                   "executable": spec.get("executable") is True, "missing": spec.get("missing")}
            outputs["launch_specs"].append(row)
            if not row["executable"]:
                not_executable.append(row)
        payload = {"workflow": "fs_plan", "steps": steps, "outputs": outputs,
                   "verdict": "RED" if not_executable else "PASS",
                   "show_user": ("Read each plan_hash below and type it back to /fs-launch with its launch_spec; "
                                 "no hash is ever filled in for you.") if not not_executable else None}
        if not_executable:
            payload["red_step"] = {"reason": "not executable", "specs": not_executable}
        ref = write_artifact(Artifact("workflow_record", "fs_plan", payload, provenance, 1),
                             request.get("out") or ctx.artifacts_dir)
        return SkillResult(status=Status.PASS, payload=payload, artifacts=(ref,), findings=(),
                           next_actions=(), provenance=provenance)

    def _stop(self, result: SkillResult, steps: list[dict[str, Any]], outputs: dict[str, Any],
              request: dict[str, Any], ctx: SkillContext, provenance: Any) -> SkillResult | None:
        """None to continue; otherwise the mapped result for the first non-PASS composed step."""
        cls = outcome_class(result)
        if cls == "ok":
            return None
        failed = steps[-1]["skill"]
        if cls == "refused":
            finding = self.finding("FP-IN-003", f"the composed {failed} refused: {result.refusal or 'see its findings'}",
                                   {"steps": steps}, f"fix what {failed}'s findings name, {_BACK}")
            return SkillResult(status=Status.REFUSED, payload={}, findings=(finding,), refusal=finding.message,
                               provenance=provenance)
        payload = {"workflow": "fs_plan", "steps": steps, "outputs": outputs,
                   "verdict": "RED" if cls == "red" else "UNMEASURED", "show_user": None}
        if cls == "red":
            payload["red_step"] = {"reason": f"{failed} came back RED"}
        ref = write_artifact(Artifact("workflow_record", "fs_plan", payload, provenance, 1),
                             request.get("out") or ctx.artifacts_dir)
        if cls == "red":  # FP-HO-001 fires in check_handoff from payload["red_step"]
            return SkillResult(status=Status.PASS, payload=payload, artifacts=(ref,), findings=(), provenance=provenance)
        finding = self.finding("FP-HO-002", f"the composed {failed} came back UNMEASURED; the plan is not a pass",
                               {"steps": steps}, f"resolve what {failed} could not measure, {_BACK}")
        return SkillResult(status=Status.UNMEASURED, payload=payload, artifacts=(ref,), findings=(finding,),
                           provenance=provenance)

    # -- handoff -----------------------------------------------------------------------------------
    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        red = (result.payload or {}).get("red_step")
        if not red:
            return []
        return [self.finding("FP-HO-001", f"the planning workflow cannot PASS: {red.get('reason')}",
                             {"red_step": red, "steps": result.payload.get("steps", [])},
                             f"fix the RED step or the spec's `missing` item, {_BACK}")]

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        return Diagnosis(
            symptom=str(getattr(failure, "refusal", None) or failure),
            likely_causes=("a missing goal or probe report", "a composed step that did not PASS",
                           "an emitted spec with executable false"),
            checks=("read the workflow_record steps: the first non-PASS step names its own findings",),
            recovery=("fix what that step names and re-run /fs-plan",),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        return {
            "FP-IN-001": {"request": {"probe_report": "probe.json"},
                          "expected": "REFUSED 96 naming FP-IN-001; nothing planned"},
            "FP-IN-002": {"request": {"goal": "goal_spec.json"},
                          "expected": "REFUSED 96 naming FP-IN-002 (probe first); nothing planned"},
            "FP-IN-003": {"request": {"goal": "goal_spec.json", "probe_report": "probe.json"},
                          "simulated_step": "training.planner refuses (the goal names no known model)",
                          "expected": "REFUSED 96 naming FP-IN-003 with the planner's findings as evidence steps"},
            "FP-HO-001": {"request": {"goal": "goal_spec.json", "probe_report": "probe.json", "dataset": "dataset.json",
                                      "model": "/models/m"},
                          "simulated_step": "training.emit returns a spec with executable false (missing: sft objective)",
                          "expected": "RED 5 naming FP-HO-001; the workflow_record names the spec's missing item"},
            "FP-HO-002": {"request": {"goal": "goal_spec.json", "probe_report": "probe.json"},
                          "simulated_step": "probe_first returns UNMEASURED (the probe was not --deep)",
                          "expected": "UNMEASURED 95 naming FP-HO-002; never PASS"},
        }
