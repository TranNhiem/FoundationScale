
"""``training.emit`` -- turn a training_plan into fs_launch_spec artifacts.

One spec per plan stage via ``emit_train`` (train stages) or ``emit_rl``
(rl stages); the skill never launches. Launching is a separate,
confirmation-gated step (``interfaces.fs.launch``).

Choices where the spec is silent:

* The request carries either one ``dataset`` payload (applied to every stage)
  or a ``datasets`` map from stage name to payload. Every stage needs a
  dataset (FS has no dataset-less stage), so a stage without one is FS-IN-002.
* ``output_dir`` per stage is ``<output_root>/<run_prefix>-<stage_name>``.
* An sbatch is rendered for train stages when the target is a cluster: nodes
  > 1, or a hardware_id naming a scheduler-managed estate (anything but
  ""/local/local-single-node). RL specs stay sbatch-less: FS RLTrainer is
  single-device, so there is nothing to schedule across nodes.
* Capabilities come from ``ctx.capabilities``; a fresh ``probe()`` is run
  only when the context does not carry one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from foundationskills.core import (
    Artifact,
    BaseSkill,
    Diagnosis,
    Finding,
    FSInterface,
    Scope,
    Severity,
    RuleSpec,
    SkillContext,
    SkillResult,
    Status,
    make_provenance,
    write_artifact,
)
from foundationskills.interfaces.fs.emit_rl import emit_rl
from foundationskills.interfaces.fs.emit_train import emit_train
from foundationskills.interfaces.fs.sbatch import render_sbatch, write_sbatch

_LOCAL_HARDWARE_IDS = {"", "local", "local-single-node", "none"}


def _capabilities(ctx: SkillContext) -> Any:
    caps = getattr(ctx, "capabilities", None)
    if caps is None:
        from foundationskills.interfaces.fs.capabilities import probe

        caps = probe()
    return caps


class TrainingEmitSkill(BaseSkill):
    """Emit one fs_launch_spec artifact per training-plan stage."""

    name = "training.emit"
    version = "0.1.0"
    description = (
        "Emit FS launch specs (and sbatch scripts where a scheduler applies) "
        "from a training_plan; never launches."
    )
    scope = Scope(
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "dgx-h100", "local"),
    )
    consumes = ("training_plan", "dataset")
    produces = ("fs_launch_spec",)

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["plan", "model", "output_root"],
        "additionalProperties": True,
        "properties": {
            "plan": {"type": "object"},
            "dataset": {"type": "object"},
            "datasets": {"type": "object"},
            "model": {"type": "string", "minLength": 1},
            "output_root": {"type": "string", "minLength": 1},
            "nodes": {"type": "integer", "minimum": 1},
            "gpus_per_node": {"type": "integer", "minimum": 1},
            "hardware_id": {"type": "string"},
            "run_prefix": {"type": "string"},
        },
    }
    output_schema: dict[str, Any] = {
        "type": "object",
        "required": ["specs"],
        "additionalProperties": True,
        "properties": {"specs": {"type": "array"}},
    }
    rules = (
        RuleSpec("FS-IN-001", "plan has no stages", Severity.BLOCK, "input"),
        RuleSpec("FS-IN-002", "no dataset for a stage that needs one", Severity.BLOCK, "input"),
        RuleSpec("FS-IN-003", "FS not importable (capabilities.available is false)", Severity.BLOCK, "input"),
        RuleSpec("FS-HO-001", "a launch spec is non-executable", Severity.WARN, "handoff"),
        RuleSpec("FS-HO-002", "all launch specs are non-executable", Severity.BLOCK, "handoff"),
        RuleSpec("FS-HO-003", "an RL spec is single-device (FS RLTrainer runs on one device)", Severity.INFO, "handoff"),
    )
    fs_interface = FSInterface(
        entries=("foundationscale-train", "fskills-rl"),
        emits=("fs_launch_spec", "sbatch"),
        apis=("emit_train", "emit_rl", "render_sbatch"),
        notes="Emission only; launching is confirmation-gated in interfaces.fs.launch.",
    )

    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        caps = _capabilities(ctx)
        if not getattr(caps, "available", False):
            findings.append(
                self.finding(
                    "FS-IN-003",
                    "missing: importable foundationscale",
                    {"probe_errors": list(getattr(caps, "errors", ()) or ())},
                    "install foundationscale, then re-run the probe",
                )
            )
        plan = request.get("plan") or {}
        stages = plan.get("stages") or []
        if not stages:
            findings.append(
                self.finding(
                    "FS-IN-001",
                    "plan has no stages",
                    {"plan_keys": sorted(plan.keys())},
                    "produce a training_plan with at least one stage before emitting",
                )
            )
            return findings
        datasets_map = request.get("datasets") or {}
        single = request.get("dataset")
        for stage in stages:
            name = str(stage.get("name") or "<unnamed>")
            if single is None and name not in datasets_map:
                findings.append(
                    self.finding(
                        "FS-IN-002",
                        f"stage {name!r} has no dataset payload (neither request.dataset nor datasets[{name!r}])",
                        {"stage": name, "stage_kind": stage.get("stage")},
                        "attach the dataset artifact produced by the data_engine skill",
                    )
                )
        return findings

    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        caps = _capabilities(ctx)
        plan = request["plan"]
        stages = plan["stages"]
        model = str(request["model"])
        output_root = Path(str(request["output_root"]))
        nodes = int(request.get("nodes") or 1)
        gpus_per_node = int(request.get("gpus_per_node") or 1)
        hardware_id = str(request.get("hardware_id") or "")
        run_prefix = str(request.get("run_prefix") or "fskills")
        datasets_map = request.get("datasets") or {}
        single_dataset = request.get("dataset")

        launch_dir = ctx.workdir / "launch"
        launch_dir.mkdir(parents=True, exist_ok=True)

        specs: list[dict[str, Any]] = []
        artifact_refs = []
        for stage in stages:
            stage_name = str(stage.get("name") or "stage")
            run_name = f"{run_prefix}-{stage_name}"
            output_dir = str(output_root / run_name)
            dataset = datasets_map.get(stage_name) or single_dataset or {}
            hardware = {"id": hardware_id, "scheduler": _guess_scheduler(hardware_id, nodes)}

            if str(stage.get("stage")) == "rl":
                spec = emit_rl(
                    stage, dataset=dataset, model=model, output_dir=output_dir,
                    caps=caps, run_name=run_name,
                )
            else:
                spec = emit_train(
                    stage, dataset=dataset, model=model, output_dir=output_dir,
                    hardware=hardware, nodes=nodes, gpus_per_node=gpus_per_node,
                    caps=caps, run_name=run_name,
                )
                # Sbatch layering: only for train stages on a cluster target;
                # launcher matches the argv emit_train produced.
                if nodes > 1 or hardware_id.lower() not in _LOCAL_HARDWARE_IDS:
                    world = nodes * gpus_per_node
                    launcher = "torchrun" if world > 1 else "python"
                    sbatch_text = render_sbatch(
                        spec["argv"],
                        env=spec["env"],
                        hardware_id=hardware_id,
                        nodes=nodes,
                        gpus_per_node=gpus_per_node,
                        job_name=run_name,
                        log_dir=str(launch_dir),
                        launcher=launcher,
                    )
                    spec = {**spec, "sbatch": sbatch_text}
                    write_sbatch(launch_dir / f"{run_name}.sbatch", sbatch_text)

            artifact = Artifact(
                type="fs_launch_spec",
                id=run_name,
                payload=spec,
                provenance=make_provenance(
                    self.name, self.version, {"training_plan": "<request>", "dataset": "<request>"}
                ),
            )
            artifact_refs.append(write_artifact(artifact, launch_dir))
            specs.append(spec)

        return SkillResult(
            status=Status.PASS,
            payload={"specs": specs, "launch_dir": str(launch_dir)},
            artifacts=tuple(artifact_refs),
        )

    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        specs = result.payload.get("specs") or []
        non_executable = [s for s in specs if s.get("executable") is not True]
        for spec in non_executable:
            findings.append(
                self.finding(
                    "FS-HO-001",
                    f"launch spec {spec.get('stage_name')!r} is non-executable: {spec.get('missing')}",
                    {"stage_name": spec.get("stage_name"), "missing": spec.get("missing")},
                    "resolve the named FS capability gap or adjust the plan stage",
                )
            )
        if specs and len(non_executable) == len(specs):
            findings.append(
                self.finding(
                    "FS-HO-002",
                    f"all {len(specs)} launch spec(s) are non-executable; nothing here can be launched",
                    {"stage_names": [s.get("stage_name") for s in specs]},
                    "install/meet the missing FS capabilities and re-emit",
                )
            )
        for spec in specs:
            if spec.get("entry") == "fskills-rl":
                findings.append(
                    self.finding(
                        "FS-HO-003",
                        f"RL spec {spec.get('stage_name')!r} is single-device: FS RLTrainer runs on one GPU",
                        {"stage_name": spec.get("stage_name")},
                        "size the RL stage for a single device (see method selection)",
                    )
                )
        return findings

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (
            f"{type(failure).__name__}: {failure}"
            if isinstance(failure, BaseException)
            else f"training.emit returned {failure.status.value}: {failure.refusal or 'see findings'}"
        )
        return Diagnosis(
            symptom=symptom,
            likely_causes=(
                "installed FS capabilities changed since the plan was made",
                "stage hyperparameters request an axis FS refuses (pp/ep > 1)",
                "a stage's dataset payload has no shards",
            ),
            checks=(
                "fskills probe --deep and compare against the plan's assumptions",
                "inspect the specs' missing fields under workdir/launch",
            ),
            recovery=(
                "re-run the planner against fresh capabilities",
                "hand data preparation back to the data_engine skill",
            ),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        """One negative fixture per rule.

        Capability-dependent fixtures carry a ``capabilities`` note: the
        harness builds an FSCapabilities with those fields overridden (e.g.
        available=False for FS-IN-003, refused pp for the handoff fixtures,
        runnable dr_grpo for FS-HO-003).
        """
        good_stage = {"name": "sft1", "stage": "sft", "algorithm": None, "method": "full", "hparams": {}}
        pp_stage = {"name": "bad", "stage": "sft", "algorithm": None, "method": "full", "hparams": {"pp": 2}}
        rl_stage = {"name": "rl1", "stage": "rl", "algorithm": "dr_grpo", "method": "lora", "hparams": {}}
        dataset = {
            "format": "sft",
            "shards": [{"path": "ds/shard-00000.jsonl"}],
            "fs_columns": {"text_column": "text", "image_column": None, "gold_key": "answer"},
        }
        return {
            "FS-IN-001": {
                "request": {"plan": {"stages": []}, "model": "m", "output_root": "out"},
                "expected": "REFUSED naming FS-IN-001",
            },
            "FS-IN-002": {
                "request": {"plan": {"stages": [good_stage]}, "model": "m", "output_root": "out"},
                "expected": "REFUSED naming FS-IN-002 (stage sft1 has no dataset)",
            },
            "FS-IN-003": {
                "request": {"plan": {"stages": [good_stage]}, "dataset": dataset, "model": "m", "output_root": "out"},
                "capabilities": {"available": False},
                "expected": "REFUSED with 'missing: importable foundationscale'",
            },
            "FS-HO-001": {
                "request": {
                    "plan": {"stages": [good_stage, pp_stage]},
                    "dataset": dataset, "model": "m", "output_root": "out",
                },
                "capabilities": {"refused_axes": ("pp", "ep")},
                "expected": "PASS carrying a WARN FS-HO-001 for stage 'bad'",
            },
            "FS-HO-002": {
                "request": {
                    "plan": {"stages": [pp_stage]},
                    "dataset": dataset, "model": "m", "output_root": "out",
                },
                "capabilities": {"refused_axes": ("pp", "ep")},
                "expected": "RED via handoff BLOCK (all specs non-executable)",
            },
            "FS-HO-003": {
                "request": {
                    "plan": {"stages": [rl_stage]},
                    "dataset": {**dataset, "format": "rl"},
                    "model": "m", "output_root": "out",
                },
                "capabilities": {"rl_runnable": {"dr_grpo": None}},
                "expected": "PASS carrying an INFO FS-HO-003 (single-device RL)",
            },
        }


def _guess_scheduler(hardware_id: str, nodes: int) -> str | None:
    """Best-effort scheduler for profile/sbatch decisions; None for local."""
    if nodes > 1 or hardware_id.lower() not in _LOCAL_HARDWARE_IDS:
        return "slurm"
    return None
