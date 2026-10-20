
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

from collections.abc import Mapping
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
from foundationskills.interfaces.fs.agentic_probe import probe_agentic, runtime_from_dict
from foundationskills.interfaces.fs.emit_agentic import emit_agentic
from foundationskills.interfaces.fs.emit_rl import emit_rl
from foundationskills.interfaces.fs.emit_train import emit_train
from foundationskills.interfaces.fs.sbatch import render_sbatch, write_sbatch
from foundationskills.interfaces.fs.video_envelope import beyond_envelope, frame_envelope

_LOCAL_HARDWARE_IDS = {"", "local", "local-single-node", "none"}


def _capabilities(ctx: SkillContext) -> Any:
    caps = getattr(ctx, "capabilities", None)
    if caps is None:
        from foundationskills.interfaces.fs.capabilities import probe

        caps = probe()
    return caps


def _agentic_fs_src() -> Any:
    """The FS source dir handed to the agentic probe/emitter (``None`` without FS).

    A4.2: the agentic probe and FS's ``--dry-run`` run INSIDE the container the stage
    declares, over this checkout (``enroot start --mount $HOME:$HOME``); the same helper
    lives in the planner skill (``TrainingPlannerSkill._agentic_fs_src``).
    """
    try:
        import foundationscale

        return Path(foundationscale.__file__).resolve().parents[1]  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - the probe reports what is missing; never assumed
        return None


def _stage_runtime(stage: Mapping[str, Any]) -> Any:
    """The ``Runtime`` an agentic stage declares (empty when the record omits one)."""
    raw = stage.get("runtime")
    if raw is None and isinstance(stage.get("agentic"), Mapping):
        raw = stage["agentic"].get("runtime")
    return runtime_from_dict(raw or {})


def _cap_field(caps: Any, name: str, default: Any = None) -> Any:
    """Read one probe field from a mapping- or attribute-shaped probe result."""
    if isinstance(caps, Mapping):
        return caps.get(name, default)
    return getattr(caps, name, default)


def _is_agentic_spec(spec: Mapping[str, Any]) -> bool:
    """A spec ``emit_agentic`` produced (FS's ``agentic_rl`` config), by any marker."""
    if str(spec.get("stage") or spec.get("stage_kind") or "") == "agentic_rl":
        return True
    if "agentic" in str(spec.get("entry") or ""):
        return True
    return "foundationscale.agentic_rl" in [str(a) for a in spec.get("argv") or ()]


def _dry_run_record(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """The FS ``--dry-run`` record an agentic spec reports, at every spelling.

    ``emit_agentic`` reports the validation it ran inside the declared runtime
    (``_DRY_RUN_TIMEOUT_S``-bounded); spellings vary across emitters and an absent
    record is returned as ``{}`` -- the caller refuses a spec it cannot read, because an
    unmeasured ``--dry-run`` is never a pass (FS-HO-005: FS owns the schema).
    """
    for key in ("fs_dry_run", "dry_run", "fs_validation", "validation"):
        record = spec.get(key)
        if isinstance(record, Mapping):
            return record
    for key in ("fs_dry_run_rc", "dry_run_rc", "fs_validation_rc", "validation_rc"):
        if spec.get(key) is not None:
            return {"rc": spec.get(key)}
    return {}


class TrainingEmitSkill(BaseSkill):
    """Emit one fs_launch_spec artifact per training-plan stage."""

    requires_fs = True  # A1.5: plans and specs are measured against the installed FS
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
        RuleSpec("FS-IN-004", "a stage declares a modality the dataset's fs_columns cannot feed", Severity.BLOCK, "input"),
        RuleSpec("FS-HO-001", "a launch spec is non-executable", Severity.WARN, "handoff"),
        RuleSpec("FS-HO-002", "all launch specs are non-executable", Severity.BLOCK, "handoff"),
        RuleSpec("FS-HO-003", "an RL spec is single-device (FS RLTrainer runs on one device)", Severity.INFO, "handoff"),
        RuleSpec("FS-HO-004", "the emitted spec lacks --fused-loss while the stage declares it", Severity.BLOCK, "handoff"),
        # A4.4 (video frame budget): the frame count is the declaration FS refuses video rows without,
        # and a count past the measured envelope (interfaces.fs.video_envelope) is UNMEASURED.
        RuleSpec("FS-IN-006", "a video stage declares no positive-integer video_frames (stage data.fs_columns or "
                 "dataset fs_columns): FS refuses video rows without FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES",
                 Severity.BLOCK, "input"),
        RuleSpec("FS-HO-007", "an emitted spec's video_frames lies above the measured frame envelope (16 uniform "
                 "frames per clip): UNMEASURED, never PASS", Severity.WARN, "handoff"),
        # A4.2 (agentic RL): the agentic spec is FS's own AgenticRLConfig, proven by FS's
        # own --dry-run INSIDE the declared runtime (FS owns the schema end to end).
        RuleSpec(
            "FS-IN-005",
            "an agentic stage whose declared runtime's probe is UNMEASURED or whose engine is not importable "
            "there (the refusal names the container): the agentic stack has no host vLLM",
            Severity.BLOCK,
            "input",
        ),
        RuleSpec(
            "FS-HO-005",
            "FS --dry-run of the generated agentic config does not exit 0 (or reports no record): RED",
            Severity.BLOCK,
            "handoff",
        ),
        RuleSpec(
            "FS-HO-006",
            "a launch/claim record maps TIMEOUT/STEP_LIMIT to abstention, or claims PASS over 0 measured "
            "steps: REFUSED (a check over spec['claims'] when present, never fired without one)",
            Severity.BLOCK,
            "handoff",
        ),
    )
    fs_interface = FSInterface(
        entries=("foundationscale-train", "fskills-rl"),
        emits=("fs_launch_spec", "sbatch"),
        apis=("emit_train", "emit_rl", "render_sbatch"),
        notes="Emission only; launching is confirmation-gated in interfaces.fs.launch.",
    )

    def __init__(self, agentic_probe: Any = probe_agentic) -> None:
        # A4.2: the agentic capability probe is injectable exactly as in the planner skill
        # (TrainingPlannerSkill.__init__): a MUST_FIRE fixture feeds its ``agentic_caps``
        # through a fake probe and never needs a real container.
        super().__init__()
        self.agentic_probe = agentic_probe

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
            if str(stage.get("stage")) == "agentic_rl":
                # A4.2: an agentic stage is fed from rollout.tasks_path inside the declared
                # runtime (emit_agentic takes no dataset payload) -- the "stage that needs
                # one" is a train/rl/preference stage.
                continue
            if single is None and name not in datasets_map:
                findings.append(
                    self.finding(
                        "FS-IN-002",
                        f"stage {name!r} has no dataset payload (neither request.dataset nor datasets[{name!r}])",
                        {"stage": name, "stage_kind": stage.get("stage")},
                        "attach the dataset artifact produced by the data_engine skill",
                    )
                )
        for stage in stages:
            name = str(stage.get("name") or "<unnamed>")
            modalities = set(stage.get("modalities") or ())
            if not modalities or str(stage.get("stage")) == "agentic_rl":
                # A4.2: an agentic stage is fed by rollout.tasks_path inside the declared
                # runtime, not by dataset fs_columns (there is no modality column to feed).
                continue
            stage_dataset = datasets_map.get(name) or single or {}
            fs_cols = dict(stage_dataset.get("fs_columns") or {})
            _modality_columns = {"text": "text_column", "image": "image_column", "video": "video_column"}
            for modality in sorted(modalities):
                col_key = _modality_columns.get(modality)
                if col_key is None:
                    continue
                col_val = fs_cols.get(col_key) or fs_cols.get(col_key.replace("_column", "_columns"))
                if modality == "text" and not col_val:
                    # conversation rows feed text through the conversations column
                    # (examples/vlm-2xH200/scripts/train_sft.sh:13), not a flat text column
                    col_val = fs_cols.get("conversations_column")
                if modality == "video" and col_val:
                    frames = _stage_video_frames(stage, stage_dataset)
                    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
                        findings.append(
                            self.finding(
                                "FS-IN-006",
                                f"stage {name!r} feeds video but video_frames is {frames!r}: FS refuses "
                                "video rows without FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES",
                                {"stage": name, "fs_columns": dict(fs_cols),
                                 "stage_fs_columns": dict((stage.get("data") or {}).get("fs_columns") or {})},
                                "declare video_frames (a positive integer, e.g. 16) in the plan stage's "
                                "data.fs_columns or the dataset's fs_columns",
                            )
                        )
                if not col_val:
                    findings.append(
                        self.finding(
                            "FS-IN-004",
                            f"stage {name!r} declares modality {modality!r} but the dataset's "
                            f"fs_columns {col_key} is missing or empty",
                            {"stage": name, "modality": modality, "fs_columns": dict(fs_cols)},
                            f"attach a dataset with {col_key} set, or drop {modality!r} from the stage's modalities",
                        )
                    )
        # A4.2 (FS-IN-005): an agentic stage is emitted only over a runtime whose probe
        # measured its engine importable THERE -- the agentic stack runs inside the enroot
        # container (vllmsrv3: python3 -m foundationscale.agentic_rl) and the host has no
        # vLLM, so a host probe is not the measurement. The refusal names the container.
        for stage in stages:
            if str(stage.get("stage")) != "agentic_rl":
                continue
            name = str(stage.get("name") or "<unnamed>")
            runtime = _stage_runtime(stage)
            agentic_caps = self.agentic_probe(runtime, fs_src=_agentic_fs_src())
            engines = _cap_field(agentic_caps, "engines_importable")
            engines_map = dict(engines) if isinstance(engines, Mapping) else {}
            engine_decl = stage.get("engine") if isinstance(stage.get("engine"), Mapping) else {}
            engine_kind = str(engine_decl.get("kind") or "vllm")
            container = str(getattr(runtime, "container", "") or "")
            if not isinstance(engines, Mapping) or engines_map.get(engine_kind) is not True:
                findings.append(
                    self.finding(
                        "FS-IN-005",
                        f"agentic stage {name!r}: engine {engine_kind!r} is not measurable as importable in "
                        f"runtime {container or '<declared runtime>'!r} "
                        f"(probe engines_importable={engines_map or 'UNMEASURED'}); the agentic stack has no "
                        f"host vLLM, and a host probe is not the measurement",
                        {"stage": name, "engine": engine_kind, "runtime_probe": agentic_caps},
                        f"fix the declared runtime ({container or 'see stage.runtime'}: the enroot "
                        f"container/mounts/env) so the in-runtime probe reports {engine_kind!r} importable, "
                        f"then re-probe before emitting",
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

        # The hardware profile carries launch facts measured on the cluster
        # (GB200 NCCL pins, CPUs per task, device peak for FS's MFU). Emitting
        # with only an id silently dropped all of them.
        hardware: dict[str, Any] = {"id": hardware_id, "scheduler": _guess_scheduler(hardware_id, nodes)}
        try:
            from foundationskills.skills.training.knowledge import load_hardware
            from foundationskills.skills.training.planner import _resolve_hardware

            # same id resolution as the planner: "gb200" -> gb200-189gb. An exact-id lookup missed it
            # (found 2026-10-07): the GB200 NCCL pins (bond0) were dropped, FS fell back to its
            # profile's eth0, which these trays do not have, and the 4-rank fabric probe refused 96.
            profile, _resolved_id = _resolve_hardware(load_hardware(), hardware_id, None)
            if profile is not None:
                raw = dict(getattr(profile, "raw", None) or {})
                hardware = {**raw, **{k: getattr(profile, k) for k in ("env", "cpus_per_task", "bf16_dense_tflops",
                                                                    "peak_provenance", "scheduler")
                                      if hasattr(profile, k)}, "id": hardware_id}
                hardware.setdefault("scheduler", _guess_scheduler(hardware_id, nodes))
        except Exception:  # noqa: BLE001 - emission proceeds; the spec notes what is missing
            pass

        specs: list[dict[str, Any]] = []
        artifact_refs = []
        previous: dict[str, Any] | None = None
        for stage in stages:
            stage_name = str(stage.get("name") or "stage")
            run_name = f"{run_prefix}-{stage_name}"
            output_dir = str(output_root / run_name)
            dataset = datasets_map.get(stage_name) or single_dataset or {}
            # A stage starts from the previous stage's output, not the base.
            stage_model, model_source, chain_missing = model, "base", None
            if previous is not None:
                model_source = f"stage:{previous['name']}"
                if previous["kind"] == "rl":
                    chain_missing = (f"missing: stage {previous['name']!r} (RL) produced no checkpoint "
                                     f"(FS RLTrainer does not save)")
                elif previous["method"] in ("lora", "qlora"):
                    chain_missing = (f"missing: merge the {previous['name']!r} LoRA adapter into the base model "
                                     f"(FS writes adapter-only checkpoints): `fskills merge plan --base <base> --adapter "
                                     f"{previous['output_dir']}/final --out <dir>`, launch it, then `fskills merge verify`")
                stage_model = f"{previous['output_dir']}/final"

            if str(stage.get("stage")) == "agentic_rl":
                # A4.2: emit_agentic renders FS's own AgenticRLConfig and proves it with FS's
                # --dry-run INSIDE the declared runtime (FS-HO-005: FS owns the schema, so a
                # key this line wrote that foundationscale.agentic_rl.config does not declare
                # would make the whole spec worthless). No dataset and no sbatch: rollout.
                # tasks_path feeds the stage, and the enroot envelope in the emitted argv IS
                # the launch wrap the measured launcher uses (run.sh: enroot start --rw ...).
                spec = emit_agentic(
                    stage,
                    output_dir=output_dir,
                    run_name=run_name,
                    fs_src=_agentic_fs_src(),
                    agentic_caps=self.agentic_probe(_stage_runtime(stage), fs_src=_agentic_fs_src()),
                )
            elif str(stage.get("stage")) in ("rl", "preference"):  # FS RLTrainer / PreferenceTrainer via fskills-rl
                spec = emit_rl(
                    stage, dataset=dataset, model=stage_model, output_dir=output_dir,
                    caps=caps, run_name=run_name,
                )
                if hardware_id.lower() not in _LOCAL_HARDWARE_IDS:
                    # RL is single-device, but on a cluster target it still needs a Slurm job:
                    # without an sbatch, `fskills launch` ran the argv on whatever host it was on
                    # (found 2026-10-07; a login node has no GPU). One task, one GPU.
                    sbatch_text = render_sbatch(
                        spec["argv"], env={**dict(hardware.get("env") or {}), **dict(spec.get("env") or {})},
                        hardware_id=hardware_id, nodes=1, gpus_per_node=1, job_name=run_name,
                        log_dir=str(Path(launch_dir).resolve()), launcher="python", cwd=spec.get("cwd"),
                        cpus_per_task=hardware.get("cpus_per_task"),
                    )
                    spec = {**spec, "sbatch": sbatch_text}
                    write_sbatch(launch_dir / f"{run_name}.sbatch", sbatch_text)
            else:
                spec = emit_train(
                    stage, dataset=dataset, model=stage_model, output_dir=output_dir,
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
                        log_dir=str(Path(launch_dir).resolve()),
                        launcher=launcher,
                        cwd=spec.get("cwd"),
                        cpus_per_task=hardware.get("cpus_per_task"),
                    )
                    spec = {**spec, "sbatch": sbatch_text}
                    write_sbatch(launch_dir / f"{run_name}.sbatch", sbatch_text)

            spec = {**spec, "model_source": model_source,
                    "launch_target": "local" if hardware_id.lower() in _LOCAL_HARDWARE_IDS and nodes == 1 else "slurm"}
            if chain_missing:
                spec = {**spec, "executable": False,
                        "missing": "; ".join(m for m in (spec.get("missing"), chain_missing) if m)}
            # A stage the planner judged infeasible (e.g. memory > GPU) is never
            # executable: launching it only buys a measured OOM (measured 2026-09-27).
            feas = stage.get("feasibility") or {}
            if feas.get("verdict") == "infeasible":
                blocked = [str(f.get("detail")) for f in feas.get("findings") or []
                           if isinstance(f, dict) and f.get("passed") is False]
                reason = "plan infeasible: " + ("; ".join(blocked) or "stage feasibility verdict is infeasible")
                spec = {**spec, "executable": False,
                        "missing": "; ".join(m for m in (spec.get("missing"), reason) if m)}
            spec = {**spec, "stage_fused_loss": str((stage.get("hparams") or {}).get("fused_loss") or "")}
            # A4.2 (FS-HO-006): a launch/claim record handed in for this stage travels WITH
            # the spec so .check_handoff can judge the claim. emit_agentic itself writes no
            # score and no verdict (the claim is made where the launch record is read:
            # rl_driver/CLI), and a spec without claims is never refused.
            claims = stage.get("claims") or request.get("claims")
            if claims:
                spec = {**spec, "claims": list(claims) if isinstance(claims, (list, tuple)) else [claims]}
            previous = {"name": stage_name, "kind": str(stage.get("stage")),
                        "method": str(stage.get("method") or "full"), "output_dir": output_dir}
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

        result = SkillResult(
            status=Status.PASS,
            payload={"specs": specs, "launch_dir": str(launch_dir)},
            artifacts=tuple(artifact_refs),
        )
        beyond = []
        for stage in stages:
            if "video" not in set(stage.get("modalities") or ()):
                continue
            stage_dataset = datasets_map.get(str(stage.get("name"))) or single_dataset or {}
            reason = beyond_envelope(_stage_video_frames(stage, stage_dataset))
            if reason:
                beyond.append((str(stage.get("name") or "<unnamed>"), reason))
        if not beyond:
            return result
        # A4.4 (FS-HO-007): a spec at a frame count no run has trained at is UNMEASURED, never PASS.
        # The handoff checks still run here (the runner skips check_handoff off PASS); a BLOCK
        # among them is a measured failure and outranks the unmeasured envelope: RED.
        handoff = list(self.check_handoff(result, ctx) or [])
        notes = [self.finding("FS-HO-007", reason, {"stage": name, "frame_envelope": frame_envelope()},
                              "emit within the measured envelope, or measure a run at this frame count first "
                              "and add it to interfaces/fs/video_envelope.py")
                 for name, reason in beyond]
        blocked = any(f.severity is Severity.BLOCK for f in handoff)
        return SkillResult(status=Status.RED if blocked else Status.UNMEASURED, payload=result.payload,
                           artifacts=result.artifacts, findings=(*handoff, *notes))

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
        for spec in specs:
            declared_fused = str(spec.get("stage_fused_loss") or "")
            if declared_fused and "--fused-loss" not in (spec.get("argv") or []):
                findings.append(
                    self.finding(
                        "FS-HO-004",
                        f"launch spec {spec.get('stage_name')!r} declares fused_loss={declared_fused!r} "
                        f"but --fused-loss is not in the emitted argv",
                        {"stage_name": spec.get("stage_name"), "declared_fused_loss": declared_fused},
                        "ensure the installed FS advertises --fused-loss, then re-emit",
                    )
                )
        # A4.2 (FS-HO-005): an agentic spec is handed off only over a MEASURED FS --dry-run
        # exit 0 of the generated config -- FS owns the schema (foundationscale.agentic_rl
        # .config), so a failed-or-unreported validation is RED, never read as a pass.
        for spec in specs:
            if not _is_agentic_spec(spec):
                continue
            dry = _dry_run_record(spec)
            rc = dry.get("rc", dry.get("returncode"))
            finished_zero = (dry.get("ok") is True) or (rc is not None and str(rc) == "0")
            if (not dry) or (not finished_zero):
                dry_text = f"did not exit 0 (record {dict(dry)!r})" if dry else "reported no record (UNMEASURED)"
                findings.append(
                    self.finding(
                        "FS-HO-005",
                        f"launch spec {spec.get('stage_name')!r}: FS --dry-run of the generated agentic config "
                        f"{dry_text}, config {spec.get('config_path') or spec.get('config')!r}",
                        {"stage_name": spec.get("stage_name"), "dry_run": dict(dry) if dry else None},
                        "fix the key FS's --dry-run names in the record and re-emit inside the declared runtime "
                        "(FS validates with --dry-run at 0 GPUs before anything launches)",
                    )
                )
        # A4.2 (FS-HO-006): claim-time honesty over whatever launch/claim record rides on a
        # spec (spec['claims']; normally ABSENT -- emit_agentic writes no score and no
        # verdict -- and "else not fired"). A terminal TIMEOUT/STEP_LIMIT laundered into
        # 'abstention', or PASS claimed over zero measured steps, is REFUSED where it is claimed.
        for spec in specs:
            raw_claims = spec.get("claims") or ()
            for claim in (list(raw_claims) if isinstance(raw_claims, (list, tuple)) else [raw_claims]):
                if not isinstance(claim, Mapping):
                    continue
                status = " ".join(str(claim.get(k) or "") for k in ("status", "outcome", "termination")).upper()
                verdict = " ".join(str(claim.get(k) or "") for k in ("claim", "verdict")).lower()
                measured = claim.get("measured_steps", claim.get("steps_measured"))
                laundered = ("TIMEOUT" in status or "STEP_LIMIT" in status) and (
                    "abstention" in verdict or "abstain" in verdict
                )
                vacuous = (not laundered) and ("pass" in verdict) and measured in (0, "0", None, "")
                if not (laundered or vacuous):
                    continue
                findings.append(
                    self.finding(
                        "FS-HO-006",
                        f"launch/claim record on spec {spec.get('stage_name')!r} "
                        + (
                            "maps a terminal TIMEOUT/STEP_LIMIT record to abstention (a non-result "
                            "laundered into an abstention)"
                            if laundered
                            else f"claims PASS over 0 measured steps (claim {dict(claim)!r})"
                        ),
                        {"stage_name": spec.get("stage_name"), "claim": dict(claim)},
                        "publish what FS adjudicated: TIMEOUT is UNMEASURED (the forced-infra arm exits 95), "
                        "a run whose steps produced no measurable update is vacuous (REFUSED, exit 96), and "
                        "PASS needs >= 1 measured step",
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

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:  # noqa: PLR0915
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
        # A4.2 (agentic): an agentic_rl stage and the runtime probe the harness feeds
        # through the injected ``agentic_probe`` (TrainingEmitSkill.__init__) -- fixtures
        # never need a real container. The model/tasks paths are deliberately unreadable:
        # FS's own --dry-run of the generated config cannot exit 0 over them.
        agentic_stage = {
            "name": "agentic0",
            "stage": "agentic_rl",
            "algorithm": "agentic_grpo_token_mean",
            "method": "full",
            "hparams": {"max_steps": 3},
            "runtime": {"kind": "host", "container": None},
            "policy": {"model_path": "/nonexistent/Qwen2.5-1.5B-Instruct"},
            "trainer": {"max_steps": 3, "lr": 1e-6, "sharding": "none"},
            "engine": {"kind": "vllm", "weight_sync_mode": "collective_rpc", "env": {"PATH": "/usr/bin:/bin"}},
            "env": {"backend": "local", "allow_unsandboxed": True},
            "harness": {"name": "qwen_xml", "parser": "qwen_xml"},
            "rollout": {"tasks_path": "/nonexistent/agentic-tasks.jsonl", "tasks_per_step": 1, "group_size": 2},
            "reward": {"kind": "music", "abc2midi_bin": "/usr/bin/abc2midi"},
        }
        agentic_caps = {
            "fs_version": "fixture",
            "algorithms": ["agentic_grpo_token_mean"],
            "engines_importable": {"vllm": True},
            "env_backends": ["local"],
            "reward_kinds": ["music"],
            "parsers": ["qwen_xml"],
            "reward_handles": {"abc2midi_bin": "/usr/bin/abc2midi"},
            "runtime": {"kind": "host"},
            "multi_gpu": None,
            "errors": [],
        }
        video_dataset = {
            "format": "mm_sft",
            "shards": [{"path": "ds/shard-00000.jsonl"}],
            "fs_columns": {"text_column": "text", "image_column": "image", "video_column": "video",
                           "gold_key": "answer"},
        }

        def video_stage(frames: int | None) -> dict[str, Any]:
            return {"name": "vid1", "stage": "sft", "algorithm": None, "method": "lora",
                    "modalities": ["image", "text", "video"],
                    "hparams": {"max_sequence_length": 16384},
                    "data": {"fs_columns": {"video_column": "video", "video_frames": frames}}}

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
            "FS-IN-004": {
                "request": {
                    "plan": {"stages": [{"name": "vlm1", "stage": "sft", "algorithm": None,
                                         "method": "lora", "modalities": ["image", "text"],
                                         "hparams": {"max_sequence_length": 16384}}]},
                    "dataset": {**dataset, "format": "mm_sft",
                                "fs_columns": {"text_column": "text", "image_column": None,
                                               "gold_key": "answer"}},
                    "model": "m", "output_root": "out",
                },
                "expected": "REFUSED naming FS-IN-004 (stage declares 'image' but dataset fs_columns cannot feed it)",
            },
            # A4.4: a video stage with no frame count anywhere is refused; 32 frames is past the
            # measured 16-frame envelope and comes back UNMEASURED.
            "FS-IN-006": {
                "request": {"plan": {"stages": [video_stage(None)]}, "dataset": video_dataset,
                            "model": "m", "output_root": "out"},
                "expected": "REFUSED naming FS-IN-006 (video stage without video_frames)",
            },
            "FS-HO-007": {
                "request": {"plan": {"stages": [video_stage(32)]}, "dataset": video_dataset,
                            "model": "m", "output_root": "out"},
                "expected": "UNMEASURED 95 naming FS-HO-007 (32 frames > the measured 16)",
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
            "FS-HO-004": {
                "request": {
                    "plan": {"stages": [{"name": "sft1", "stage": "sft", "algorithm": None, "method": "full",
                                         "hparams": {"fused_loss": "liger", "max_sequence_length": 16384}}]},
                    "dataset": dataset, "model": "m", "output_root": "out",
                },
                "capabilities": {"train_flag_choices": {"--fused-loss": ["cross_entropy"]}},
                "expected": "RED via handoff BLOCK FS-HO-004 (fused_loss=liger declared but --fused-loss not in argv)",
            },
            "FS-IN-005": {
                "request": {
                    "plan": {"stages": [{**agentic_stage, "runtime": {"kind": "enroot", "container": "vllmsrv3"}}]},
                    "model": "m", "output_root": "out",
                },
                "agentic_caps": {**agentic_caps, "engines_importable": {"vllm": False}},
                "expected": "REFUSED naming FS-IN-005 (vllm not importable in container 'vllmsrv3')",
            },
            "FS-HO-005": {
                # the host runtime runs FS's --dry-run for real (no container involved): the
                # generated config cannot resolve the /nonexistent model/tasks, so the
                # command does not exit 0 and the handoff is RED.
                "request": {"plan": {"stages": [agentic_stage]}, "model": "m", "output_root": "out"},
                "agentic_caps": agentic_caps,
                "expected": "RED via handoff BLOCK FS-HO-005 (FS --dry-run rc != 0)",
            },
            "FS-HO-006": {
                "request": {
                    "plan": {
                        "stages": [
                            {
                                **agentic_stage,
                                "claims": [
                                    {"status": "TIMEOUT", "claim": "abstention"},
                                    {"status": "STEP_LIMIT", "verdict": "PASS", "measured_steps": 0},
                                ],
                            }
                        ]
                    },
                    "model": "m", "output_root": "out",
                },
                "agentic_caps": agentic_caps,
                "expected": (
                    "RED via handoff BLOCK FS-HO-006 (TIMEOUT mapped to abstention; "
                    "PASS claimed over 0 measured steps)"
                ),
            },
        }


def _guess_scheduler(hardware_id: str, nodes: int) -> str | None:
    """Best-effort scheduler for profile/sbatch decisions; None for local."""
    if nodes > 1 or hardware_id.lower() not in _LOCAL_HARDWARE_IDS:
        return "slurm"
    return None


def _stage_video_frames(stage: Mapping[str, Any], dataset: Mapping[str, Any]) -> Any:
    """The frame count emit_train mirrors into FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES (stage first, then dataset)."""
    stage_cols = (stage.get("data") or {}).get("fs_columns") or {}
    frames = stage_cols.get("video_frames")
    return frames if frames is not None else (dataset.get("fs_columns") or {}).get("video_frames")
