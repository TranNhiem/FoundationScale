
"""TrainingPlannerSkill: goal_spec (+optional readiness) -> training_plan artifact."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from foundationskills.interfaces.fs.agentic_probe import probe_agentic, runtime_from_dict
from foundationskills.skills.training.agentic import (
    agentic_gates,
    blocking_findings,
    plan_agentic,
)

from foundationskills.core import (
    Artifact,
    BaseSkill,
    Diagnosis,
    Finding,
    FSInterface,
    RuleSpec,
    Scope,
    Severity,
    SkillContext,
    SkillResult,
    Status,
    make_provenance,
    sha256_json,
    write_artifact,
)

try:  # KnowledgeError lives in the D1-owned module; guard against import shape drift.
    from foundationskills.skills.training.knowledge import KnowledgeError
except Exception:  # pragma: no cover - defensive
    class KnowledgeError(ValueError):  # type: ignore[no-redef]
        pass

from foundationskills.interfaces.fs.capabilities import FSCapabilities, probe
from foundationskills.skills.training.planner import (
    PlanningRefusal,
    plan,
    vlm_gates,
    vlm_goal_fields,
    vlm_stage_fields,
)

TRAINING_PLAYBOOK: dict[str, Diagnosis] = {
    "loss_spike": Diagnosis(
        symptom="loss spike mid-run",
        likely_causes=(
            "a bad data shard or un-cleaned outliers",
            "LR too high for the model size / rewarm overshoot in CPT",
            "mixed-precision overflow (bf16 master weights missing)",
        ),
        checks=(
            "find the step interval and the shards it read (run manifest)",
            "compare LR schedule to the plan hparams (warmup ratio, peak LR)",
            "inspect per-source loss: drop the spiking source and resume",
        ),
        recovery=(
            "resume from the last checkpoint before the spike",
            "halve the LR or lengthen warmup, or skip the toxic shard range",
            "for CPT prefer the rewarm+redecay schedule (Ibrahim et al. 2024)",
        ),
    ),
    "divergence_nan": Diagnosis(
        symptom="loss is NaN / diverges",
        likely_causes=(
            "learning rate far too high",
            "numerics: overflow in attention/logits without gradient clipping",
            "corrupted records (empty text, non-UTF8) reaching the loss",
        ),
        checks=(
            "confirm --max-grad-norm is set (default 1.0) and the LR band matches the size",
            "grep the run logs for the first NaN step and the batch source",
            "run the clean/quality pipeline ops over the failing shard",
        ),
        recovery=(
            "restart from a clean checkpoint with a 2-5x lower LR",
            "enable gradient checkpointing/clipping; validate records numerically",
        ),
    ),
    "oom": Diagnosis(
        symptom="CUDA out of memory",
        likely_causes=(
            "per-device batch or sequence length too large for HBM",
            "optimizer state not sharded (DDP instead of FSDP) or offload off",
            "activations without gradient checkpointing",
        ),
        checks=(
            "compare the plan estimate.memory_gb_per_gpu with the GPU HBM",
            "confirm --sharding-strategy fsdp and --gradient-checkpointing",
            "pp/ep are REFUSED by FS: do not try to fix OOM with --pp/--ep",
        ),
        recovery=(
            "drop per-device batch to 1 and raise --gradient-accumulation-steps",
            "switch full->lora, or halve --max-sequence-length",
            "enable --cpu-optimizer-offload as a last resort (slower)",
        ),
    ),
    "reward_hacking": Diagnosis(
        symptom="RL reward rises while true quality falls",
        likely_causes=(
            "the verifier is exploitable (pattern-matchable answers)",
            "answer_pattern matches degenerate outputs",
            "group_size too small to separate good from lucky samples",
        ),
        checks=(
            "read raw rollouts from StepReport at rising reward steps",
            "audit RLTrainConfig.answer_pattern against the gold key format",
            "score rollouts with a held-out harder verifier",
        ),
        recovery=(
            "strengthen the verifier/answer_pattern, add length or format penalties",
            "raise group_size and diversify prompts",
        ),
    ),
    "reward_saturation_unmeasured_steps": Diagnosis(
        symptom="FS prints 'UNMEASURED step N' during RL",
        likely_causes=(
            "all advantages in the group are zero (every sample equally right/wrong)",
            "prompts too easy or too hard for the current policy",
            "temperature too low to produce diverse samples",
        ),
        checks=(
            "count consecutive UNMEASURED steps in the RL driver log",
            "inspect pass-rate spread per prompt in the StepReports",
        ),
        recovery=(
            "raise RLTrainConfig.group_size and/or temperature",
            "use prompts with intermediate difficulty (some pass, some fail)",
        ),
    ),
    "entropy_collapse": Diagnosis(
        symptom="policy entropy collapses to ~0 during RL",
        likely_causes=(
            "learning rate too high for policy-gradient updates",
            "top_p/top_k too aggressive combined with low temperature",
            "group_size too large relative to prompt diversity",
        ),
        checks=(
            "plot entropy per step from the RL driver log",
            "compare sampler settings (temperature/top_p/top_k) to the plan",
        ),
        recovery=(
            "lower the RL learning rate; raise temperature toward 1.0",
            "restore sampling diversity before continuing (checkpoint rollback)",
        ),
    ),
    "moe_router_imbalance": Diagnosis(
        symptom="MoE router imbalance (few experts take most tokens)",
        likely_causes=(
            "auxiliary load-balancing loss weight too small",
            "domain-shifted corpus over-specialises the router in CPT",
        ),
        checks=(
            "inspect per-expert token counts in the training logs",
            "check the fs family card: qwen3.5 MoE variants only",
        ),
        recovery=(
            "raise the router aux-loss weight if the family supports it",
            "increase sequence diversity/global batch so routing statistics stabilise",
        ),
    ),
    "slow_throughput": Diagnosis(
        symptom="training throughput far below the estimate",
        likely_causes=(
            "dataloader starvation (num_workers/prefetch too low)",
            "MFU over-estimated (derived from a different GPU or shape)",
            "communication-bound sharding for the model size",
        ),
        checks=(
            "measure tokens/s and compare to the plan's tokens_per_s_per_gpu*gpus",
            "watch GPU idle time vs dataloader time",
            "distinguish measured (31.8% GB200 dense) from derived/literature MFU",
        ),
        recovery=(
            "raise --dataloader-num-workers/--dataloader-prefetch-factor",
            "replan with the measured MFU for the actual hardware",
            "for small models prefer ddp over fsdp",
        ),
    ),
    "hang_no_progress": Diagnosis(
        symptom="the job is RUNNING but the log has stopped advancing",
        likely_causes=(
            "a rank died and the others wait in a collective (the scheduler still says R)",
            "a DataLoader worker died (e.g. a collate IndexError) and NCCL waits for the watchdog",
            "first-iteration compile/warm-up (can take minutes: compare with the prior run)",
        ),
        checks=(
            "judge health by GPU utilisation AND log mtime/line count, never by scheduler state",
            "grep the log for Traceback/IndexError on any rank",
            "compare elapsed time with the same config's first-iteration time in earlier runs",
        ),
        recovery=(
            "cancel only your own job id (never pkill -u on a shared account) and relaunch with the cause fixed",
            "for multimodal data: drop over-length records instead of truncating (media sentinels)",
        ),
    ),
    "lr_not_applied": Diagnosis(
        symptom="the logged learning rate differs from the plan (for example, warmup restarts after resume)",
        likely_causes=(
            "a default LR inherited from a launcher or recipe instead of the plan's explicit value",
            "a resume without the LR-scheduler state, so warmup silently re-runs",
        ),
        checks=(
            "compare the first logged learning_rate with warmup arithmetic: peak_lr * step / warmup_steps",
            "confirm --learning-rate/--warmup-steps are in the emitted argv (fskills emit always passes them)",
        ),
        recovery=("relaunch with explicit LR and warmup; resume only from checkpoints that saved the scheduler",),
    ),
    "fs_refused_96": Diagnosis(
        symptom="foundationscale-train exits 96 with [fs:train:refuse]",
        likely_causes=(
            "FS refused the launch: the missing input is named on the refuse line",
            "an axis the installed FS refuses (pp>1, ep>1) was requested",
            "a missing required flag (--profile-name/--profile-path, --model, --dataset, --output-dir)",
        ),
        checks=(
            "read the [fs:train:refuse] line; it names the missing input",
            "diff argv against the fs_launch_spec the emitter produced",
            "cross-check caps.check(stage, ...) for the refused axis/algorithm",
        ),
        recovery=(
            "fix the named input and rerun --dry-run before submitting",
            "never bypass the refusal by relaxing validation",
        ),
    ),
}


def diagnose_symptom(name: Any) -> Diagnosis:
    """Map a symptom string (or failure text) to a playbook Diagnosis."""
    norm = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    if norm in TRAINING_PLAYBOOK:
        return TRAINING_PLAYBOOK[norm]
    for key, diagnosis in TRAINING_PLAYBOOK.items():
        if key in norm or (len(norm) > 6 and norm in key):
            return diagnosis
    keyword_map = {
        "oom": ("out_of_memory", "out of memory", "cuda oom"),
        "divergence_nan": ("nan",),
        "fs_refused_96": ("refuse", "exit 96", " 96"),
        "reward_saturation_unmeasured_steps": ("unmeasured step",),
    }
    for key, needles in keyword_map.items():
        if any(needle in norm for needle in needles):
            return TRAINING_PLAYBOOK[key]
    return Diagnosis(
        symptom=f"unrecognised symptom: {name!r}",
        likely_causes=("not covered by the training playbook",),
        checks=("run TrainingPlannerSkill.diagnose with one of: " + ", ".join(sorted(TRAINING_PLAYBOOK)),),
        recovery=("escalate with the run logs and the fs_launch_spec",),
    )

# A4.1 (VLM competition LoRA): the gate rules vlm_gates() (skills/training/planner.py)
# returns, and the recovery each Finding carries. Phase mapping (shared contract):
# TR-IN-005..008 surface from .check_inputs as input findings (BLOCK -> REFUSED);
# TR-PL-008 surfaces there too (see its rule description: the handoff template can
# only emit findings, never a REFUSED result); TR-PL-006/007 surface from
# .check_handoff (BLOCK handoff -> RED).
_VLM_INPUT_RULES: tuple[str, ...] = ("TR-IN-005", "TR-IN-006", "TR-IN-007", "TR-IN-008", "TR-PL-008")
_VLM_HANDOFF_RULES: tuple[str, ...] = ("TR-PL-006", "TR-PL-007")
_VLM_RECOVERY: dict[str, str] = {
    "TR-IN-005": "register the family in FS, or pass explicit lora_targets / --adapter-target",
    "TR-IN-006": (
        "declare goal.data.fs_columns.video_column and goal.video_frames "
        "(FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN / FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES)"
    ),
    "TR-IN-007": (
        "pass explicit lora_targets / --adapter-target whose names match the model's modules "
        "(a blank-only list is not a declaration; drop the key and register the family for FS auto)"
    ),
    "TR-IN-008": (
        "plan tp=1 (FS refuses --tp wherever it cannot also save) and run cp>1 on text rows only "
        "(--cp is measured end to end only on text)"
    ),
    "TR-PL-006": (
        "plan max_sequence_length at 32768 or below (the measured VLM ceiling), or measure 128K before "
        "the plan claims it"
    ),
    "TR-PL-007": "plan max_sequence_length at 32768 or below for gemma-4-31B-it",
    "TR-PL-008": (
        "run with --fused-loss liger at max_sequence_length >= 32768 (the measured 32K loss path), or "
        "plan below 32768",
    ),
}

# A4.2 (agentic RL): the recovery each agentic gate Finding carries. Rule ids and
# severities are the agentic module's own (skills/training/agentic.py GATE_SEVERITY):
# this map never re-decides them, it only says how to recover from each refusal.
_AGENTIC_RECOVERY: dict[str, str] = {
    "TR-IN-010": (
        "run the agentic probe in the declared runtime and name only algorithm/engine/env backend/reward "
        "kind/parser entries the probe measures (an unprobed entry is UNMEASURED: register it in FS or re-probe)"
    ),
    "TR-IN-011": (
        "plan 1 GPU with trainer.sharding 'none' (the measured agentic run is single-GPU), or measure agentic "
        "multi-GPU first -- the runtime probe leaves multi_gpu UNMEASURED"
    ),
    "TR-IN-012": (
        "declare and prove every handle the scorer reads in the declared runtime (music: abc2midi_bin present "
        "and executable; rollout.tasks_path readable; the env backend named, with allow_unsandboxed when it "
        "runs at isolation none)"
    ),
    "TR-IN-013": (
        "declare a weight-sync mode on the engine, or opt in explicitly with engine.allow_stale_policy "
        "(max_steps > 1 with neither is REFUSED: staleness is an operator declaration, never a default)"
    ),
    "TR-IN-014": (
        "declare everything the engine server needs in engine.env (FS reads engine.env verbatim and never "
        "merges os.environ, so the env that starts the server is exactly what the declaration carries)"
    ),
    "TR-PL-009": (
        "agentic_grpo_token_mean is the measured control arm: the handoff names the arm as the control and "
        "keeps its GB200 measurement as the baseline other arms are compared against"
    ),
}


def _agentic_fs_src() -> Any:
    """The FS source dir the agentic probe runs over (``None`` without importable FS).

    A4.2: the probe runs INSIDE the runtime ``goal.agentic.runtime`` declares, over this
    checkout (the measured launcher mounts ``$HOME:$HOME`` into the enroot container):
    a host-side probe is not the measurement, and a missing import is UNMEASURED.
    """
    try:
        import foundationscale

        return Path(foundationscale.__file__).resolve().parents[1]  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - the probe reports what is missing; never assumed
        return None


class TrainingPlannerSkill(BaseSkill):
    """Plan an executable-or-explicitly-refused multi-stage training run."""

    requires_fs = True  # A1.5: plans and specs are measured against the installed FS
    name = "training.planner"
    version = "0.1.0"
    description = (
        "Turn a goal_spec (with an optional data readiness_report) into a training_plan "
        "whose stages are either executable on the installed FoundationScale or named "
        "with the exact missing capability."
    )
    scope = Scope(
        model_types=("llm", "vlm"),
        families="any",
        stages=("pretrain", "cpt", "sft", "preference", "rl"),
        algorithms=(),
        methods=("full", "lora", "qlora"),
        hardware=("gb200", "h100", "a100"),
    )
    consumes = ("goal_spec",)
    produces = ("training_plan",)
    input_schema = {
        "type": "object",
        "required": ["goal"],
        "additionalProperties": True,
        "properties": {
            "goal": {"type": "object"},
            "readiness": {"type": ["object", "null"]},
            "hardware_id": {"type": ["string", "null"]},
            # a stored agentic runtime probe (probe_agentic output); used only when it says measured: true,
            # else the probe runs live in the declared runtime (starting a container only to probe is slow)
            "agentic_probe": {"type": ["object", "null"]},
        },
    }
    output_schema = {
        "type": "object",
        "required": ["plan"],
        "additionalProperties": True,
        "properties": {"plan": {"type": "object"}},
    }
    rules = (
        RuleSpec("TR-IN-001", "goal.base_model missing or empty", Severity.BLOCK, "input"),
        RuleSpec("TR-IN-002", "goal.objective missing or empty", Severity.BLOCK, "input"),
        RuleSpec("TR-IN-003", "goal.hardware missing (gpu/gpus_per_node/nodes)", Severity.BLOCK, "input"),
        RuleSpec("TR-IN-004", "a RED readiness_report was supplied; fix the data first", Severity.BLOCK, "input"),
        RuleSpec(
            "TR-IN-005",
            "the VLM family is not FS-registered and no explicit lora_targets are declared (the refusal names "
            "both fixes: register the family in FS, pass explicit lora_targets / --adapter-target)",
            Severity.BLOCK,
            "input",
        ),
        RuleSpec(
            "TR-IN-006",
            "rows carry video but the feed lacks video_column or video_frames (FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN "
            "/ FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES)",
            Severity.BLOCK,
            "input",
        ),
        RuleSpec(
            "TR-IN-007",
            "explicit lora_targets empty after normalisation, or one that matches no module in the module inventory",
            Severity.BLOCK,
            "input",
        ),
        RuleSpec(
            "TR-IN-008",
            "tp>1 is refused by FS (exit 96), or cp>1 is planned with image/video rows (--cp is measured end to "
            "end only on text)",
            Severity.BLOCK,
            "input",
        ),
        RuleSpec("TR-PL-001", "plan has 0 executable stages on the installed FS", Severity.WARN, "handoff"),
        RuleSpec("TR-PL-002", "a stage is feasibility-infeasible with no executable feasible alternative", Severity.BLOCK, "handoff"),
        RuleSpec("TR-PL-003", "a stage relies on an unvalidated (literature/community/derived) recipe", Severity.INFO, "handoff"),
        RuleSpec("TR-PL-004", "a stage needs data not yet prepared (handoff to data_engine)", Severity.WARN, "handoff"),
        RuleSpec("TR-PL-005", "disclosure: FS SFT trains on the full rendered text (no assistant-only loss masking)", Severity.INFO, "handoff"),
        RuleSpec(
            "TR-PL-006",
            "planned max_sequence_length >= 131072, or a goal claim that it works (a 128K claim is a planning fact, "
            "not a measurement): marks the handoff RED",
            Severity.BLOCK,
            "handoff",
        ),
        RuleSpec(
            "TR-PL-007",
            "gemma-4-31B-it planned above max_sequence_length 32768: marks the handoff RED",
            Severity.BLOCK,
            "handoff",
        ),
        RuleSpec(
            "TR-PL-008",
            "max_sequence_length >= 32768 without fused_loss 'liger' (REFUSED, naming --fused-loss liger): an input "
            "rule by necessity, because the handoff template can only emit findings and never a REFUSED result",
            Severity.BLOCK,
            "input",
        ),
        # A4.2 (agentic RL): the TR-* gates adjudicated by skills/training/agentic.py
        # (agentic_gates) against the agentic capability probe measured INSIDE the
        # declared runtime (interfaces.fs.agentic_probe.probe_agentic).
        RuleSpec(
            "TR-IN-010",
            "algorithm/engine/env backend/reward kind/parser not probed: the agentic plan names one the declared "
            "runtime's probe does not measure (a missing or failed probe field is UNMEASURED, never assumed)",
            Severity.BLOCK,
            "input",
        ),
        RuleSpec(
            "TR-IN-011",
            "multi-GPU claimed while agentic multi-GPU is unmeasured (more than 1 GPU, or trainer.sharding != 'none')",
            Severity.BLOCK,
            "input",
        ),
        RuleSpec(
            "TR-IN-012",
            "reward handle / tasks file / env declaration missing (music: abc2midi_bin present and executable; "
            "rollout.tasks_path readable; the env backend named, allow_unsandboxed for one at isolation none)",
            Severity.BLOCK,
            "input",
        ),
        RuleSpec(
            "TR-IN-013",
            ">1 step with neither weight sync nor stale-policy opt-in (max_steps > 1 without engine weight sync "
            "and without declared engine.allow_stale_policy)",
            Severity.BLOCK,
            "input",
        ),
        RuleSpec(
            "TR-IN-014",
            "engine env cannot start the server (engine.env must be an explicit declaration: FS never merges os.environ)",
            Severity.BLOCK,
            "input",
        ),
        RuleSpec(
            "TR-PL-009",
            "the measured control chosen as the default arm (agentic_grpo_token_mean): handoff note",
            Severity.WARN,
            "handoff",
        ),
    )
    fs_interface = FSInterface(
        entries=("foundationscale-train", "fskills-rl"),
        emits=("training_plan",),
        apis=("foundationscale.train.cli", "foundationscale.rl.trainer.RLTrainer"),
        notes="This skill only plans; training.emit turns the plan into an fs_launch_spec. --dry-run runs the FS validation prologue with 0 GPUs.",
    )

    def __init__(self, agentic_probe: Any = probe_agentic) -> None:
        # A4.2: the agentic capability probe is injectable -- a MUST_FIRE fixture never
        # needs a real enroot container (the harness feeds its ``agentic_caps`` through a
        # fake probe here). Default: the in-runtime probe (agentic_probe).
        super().__init__()
        self.agentic_probe = agentic_probe

    # -- hooks -------------------------------------------------------------

    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        goal = request.get("goal") or {}
        base_model = goal.get("base_model") or {}
        if not str(base_model.get("name_or_path") or "").strip():
            findings.append(
                self.finding(
                    "TR-IN-001",
                    "goal.base_model.name_or_path is missing or empty",
                    {"base_model": base_model},
                    "name the base model (hf_id, local path, or a knowledge-base variant id)",
                )
            )
        if not str(goal.get("objective") or "").strip():
            findings.append(
                self.finding(
                    "TR-IN-002",
                    "goal.objective is missing or empty",
                    {},
                    "state what the model should be able to do after training",
                )
            )
        hardware = goal.get("hardware") or {}
        missing_hw = [k for k in ("gpu", "gpus_per_node", "nodes") if hardware.get(k) in (None, "")]
        if missing_hw:
            findings.append(
                self.finding(
                    "TR-IN-003",
                    f"goal.hardware is missing required key(s): {', '.join(missing_hw)}",
                    {"hardware": hardware},
                    "name the GPU and the cluster shape (nodes x gpus_per_node)",
                )
            )
        readiness = request.get("readiness")
        if isinstance(readiness, dict) and readiness.get("verdict") == "RED":
            findings.append(
                self.finding(
                    "TR-IN-004",
                    "a RED readiness_report was supplied: fix the data first; hand off to data_engine",
                    {"dataset_id": readiness.get("dataset_id")},
                    "rerun the data_engine skill until the readiness verdict is PASS (or address every failed check)",
                )
            )
        # A4.1 (VLM competition LoRA): the gate rules. TR-IN-005..008 surface as
        # input findings (BLOCK input -> REFUSED); TR-PL-008 does too — its subject
        # is the plan handoff (see its rule description), but the input phase is the
        # only one that can REFUSE.
        vlm_facts = vlm_goal_fields(goal)
        evidence = {
            "modalities": vlm_facts["modalities"],
            "row_modalities": vlm_facts["row_modalities"],
            "fs_columns": vlm_facts["fs_columns"],
            "fused_loss": vlm_facts["fused_loss"],
            "lora_targets": vlm_facts["lora_targets"],
            "max_sequence_length": vlm_facts["max_sequence_length"],
            "tp": vlm_facts["tp"],
            "cp": vlm_facts["cp"],
        }
        for rule_id, message in vlm_gates(vlm_stage_fields(goal), goal, getattr(ctx, "capabilities", None)):
            if rule_id in _VLM_INPUT_RULES:
                findings.append(self.finding(rule_id, message, dict(evidence), _VLM_RECOVERY[rule_id]))
        # A4.2 (agentic RL): the TR-IN-010..014 gates over the agentic stage, adjudicated
        # against the agentic capability probe measured INSIDE the runtime goal.agentic
        # declares (the agentic stack has no host vLLM: a host probe would truthfully
        # report every engine absent and every plan would read UNMEASURED). BLOCK findings
        # surface HERE as input findings (BLOCK input -> REFUSED); TR-PL-009 is a handoff
        # note and fires from .check_handoff. fs_src is the FS checkout the probe runs
        # over inside that runtime.
        if isinstance(goal.get("agentic"), dict):
            declared = dict(goal["agentic"])
            stored = request.get("agentic_probe")
            if isinstance(stored, dict) and stored.get("measured") is True:
                agentic_caps = stored
            else:
                try:
                    runtime = runtime_from_dict(declared.get("runtime") or {})
                except ValueError as exc:  # a malformed runtime is an unmeasurable one, refused by name
                    findings.append(self.finding(
                        "TR-IN-010", f"goal.agentic.runtime is malformed: {exc}",
                        {"runtime": declared.get("runtime")},
                        "declare runtime as {kind, container, mounts: [[src, dst], ...], env: [[k, v], ...], rc}"))
                    return findings
                agentic_caps = self.agentic_probe(runtime, fs_src=_agentic_fs_src())
            for rule_id, message in blocking_findings(agentic_gates(plan_agentic(goal), agentic_caps)):
                findings.append(
                    self.finding(
                        rule_id,
                        message,
                        {"agentic": declared, "runtime_probe": agentic_caps},
                        _AGENTIC_RECOVERY.get(rule_id, "run the agentic probe in the declared runtime"),
                    )
                )
        return findings

    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        goal = dict(request["goal"])
        readiness = request.get("readiness")
        caps = ctx.capabilities
        if not isinstance(caps, FSCapabilities):
            try:
                caps = probe(deep=False)
            except Exception as exc:  # noqa: BLE001 - recorded, planning degrades to non-executable
                caps = FSCapabilities(available=False, fs_version=None, errors=(f"probe failed: {type(exc).__name__}: {exc}",))
        try:
            payload = plan(goal, caps=caps, readiness=readiness, hardware_id=request.get("hardware_id"))
        except PlanningRefusal as exc:
            return self._refused(str(exc))

        artifact_id = "plan-" + sha256_json(payload)[:12]
        artifact = Artifact(
            type="training_plan",
            id=artifact_id,
            payload=payload,
            provenance=make_provenance(self.name, self.version, inputs={"goal": sha256_json(goal)[:16]}),
        )
        ref = write_artifact(artifact, ctx.artifacts_dir)
        return SkillResult(
            status=Status.PASS,
            payload={"plan": payload, "plan_ref": ref.to_dict()},
            artifacts=(ref,),
        )

    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        findings: list[Finding] = []
        plan_payload = (result.payload or {}).get("plan") or {}
        stages = plan_payload.get("stages") or []

        executable = [s for s in stages if s.get("executable")]
        if stages and not executable:
            missing_items = [f"{s.get('name')}: {s.get('missing')}" for s in stages]
            findings.append(
                self.finding(
                    "TR-PL-001",
                    "the plan has 0 executable stages on the installed FS (the plan is still useful as a gap list)",
                    {"missing": missing_items},
                    "address the missing items per stage (upgrade FS, change algorithm/axes, or shrink the ask)",
                )
            )
        for stage in stages:
            name = stage.get("name")
            feas = stage.get("feasibility") or {}
            alternatives = feas.get("alternatives") or []
            has_feasible_alt = any(
                isinstance(a, dict)
                and a.get("executable")
                and str(a.get("verdict")) in ("ok", "warn")
                for a in alternatives
            )
            if str(feas.get("verdict")) == "infeasible" and not has_feasible_alt:
                findings.append(
                    self.finding(
                        "TR-PL-002",
                        f"stage '{name}' is infeasible and has no executable feasible alternative",
                        {"stage": name, "alternatives": alternatives},
                        "shrink the model/sequence, change method, or move to larger hardware before launching",
                    )
                )
            provenance = stage.get("recipe_provenance")
            if stage.get("recipe_derived") or provenance != "validated":  # rv42: anything not validated
                findings.append(
                    self.finding(
                        "TR-PL-003",
                        f"stage '{name}' relies on an unvalidated recipe ({provenance or 'derived heuristics'})",
                        {"stage": name, "recipe_id": stage.get("recipe_id"), "provenance": provenance},
                        "treat the hparams as literature — validate on FoundationScale before a long run",
                    )
                )
            data = stage.get("data") or {}
            if data.get("handoff") == "data_engine":
                findings.append(
                    self.finding(
                        "TR-PL-004",
                        f"stage '{name}' needs data in '{data.get('target_format')}' format that is not prepared yet",
                        {"stage": name, "target_format": data.get("target_format")},
                        "hand off to the data_engine skill with the target format before launching this stage",
                    )
                )
        if any(s.get("stage") == "sft" for s in stages):
            findings.append(
                self.finding(
                    "TR-PL-005",
                    "disclosure: FS SFT trains on the full rendered `text` (no assistant-only loss masking)",
                    {"stages": [s.get("name") for s in stages if s.get("stage") == "sft"]},
                    "if assistant-only loss is required, the installed FS cannot do it; state this to the user",
                )
            )
        # A4.1 (BLOCK handoff -> RED): TR-PL-006 (a >= 128K planning claim) and
        # TR-PL-007 (gemma-4-31B-it above 32768) over the planned stages; a stage's
        # hparams carry the planned max_sequence_length the estimate priced, and the
        # plan's goal is the claim channel.
        plan_goal = plan_payload.get("goal") or {}
        for stage in stages:
            for rule_id, message in vlm_gates(stage, plan_goal, getattr(ctx, "capabilities", None)):
                if rule_id in _VLM_HANDOFF_RULES:
                    hparams = stage.get("hparams") or {}
                    findings.append(
                        self.finding(
                            rule_id,
                            message,
                            {
                                "stage": stage.get("name"),
                                "modalities": stage.get("modalities"),
                                "max_sequence_length": hparams.get("max_sequence_length"),
                                "fused_loss": hparams.get("fused_loss"),
                            },
                            _VLM_RECOVERY[rule_id],
                        )
                    )
        # A4.2 (handoff WARN): TR-PL-009 over the planned agentic stage -- when the arm is
        # agentic_grpo_token_mean the handoff must name it as the measured control it is
        # (the ablation baseline), never as "just the default". A note, not a refusal: the
        # block verdict stays with the TR-IN-* gates.
        for stage in stages:
            if str(stage.get("stage")) != "agentic_rl":
                continue
            arm = str(stage.get("algorithm") or (stage.get("hparams") or {}).get("algorithm") or "")
            if arm == "agentic_grpo_token_mean":
                findings.append(
                    self.finding(
                        "TR-PL-009",
                        "the agentic stage plans agentic_grpo_token_mean: this is the MEASURED control arm "
                        "(mean-of-group advantage on the measured GB200 music run), so the handoff carries it "
                        "as the control other arms are compared against",
                        {"stage": stage.get("name"), "algorithm": arm},
                        _AGENTIC_RECOVERY["TR-PL-009"],
                    )
                )
        return findings

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        if isinstance(failure, SkillResult):
            blob = " ".join(
                [failure.refusal or "", *(f"{f.rule_id} {f.message}" for f in failure.findings)]
            ).lower()
            if "no stage selected" in blob:
                return Diagnosis(
                    symptom="planning refused: no stage selected",
                    likely_causes=(
                        "stage_rules.yaml has no rule matching the inferred goal kind/data facts",
                        "data facts (has_pairs/answers/instructions) are absent or wrong",
                    ),
                    checks=("inspect plan decisions up to stage_selection", "check load_stage_rules() conditions against the goal"),
                    recovery=("add a stage rule, or state goal/data_facts explicitly in the goal_spec",),
                )
            return diagnose_symptom(blob)
        if isinstance(failure, PlanningRefusal):
            text = str(failure)
            if "no stage selected" in text:
                return Diagnosis(
                    symptom="planning refused: no stage selected",
                    likely_causes=("stage_rules.yaml lacks a matching rule for the goal/data facts",),
                    checks=("print the goal kinds and data facts the planner inferred",),
                    recovery=("state data_facts (has_instructions/has_verifiable_answers) or add a stage rule",),
                )
            return Diagnosis(
                symptom="planning refused: missing input",
                likely_causes=(text,),
                checks=("supply the named input; a refusal names what is missing",),
                recovery=("complete the goal_spec (see foundationskills.agent.intake.REQUIRED_FACTS)",),
            )
        if isinstance(failure, KnowledgeError) or type(failure).__name__ == "KnowledgeError":
            return Diagnosis(
                symptom="knowledge base failed to load",
                likely_causes=(
                    "a YAML file under skills/training/knowledge violates its schema",
                    "knowledge package data is missing from the install",
                ),
                checks=(f"underlying error: {failure}", "validate each knowledge YAML against schemas/knowledge/*.json"),
                recovery=("fix the named file; the loader error names it and every violation",),
            )
        return Diagnosis(
            symptom=f"planner raised {type(failure).__name__}",
            likely_causes=(
                "knowledge base missing/corrupt",
                "goal references an unknown model or hardware entry",
            ),
            checks=(f"underlying error: {failure}", "rerun planner.plan directly with a traceback",),
            recovery=("fix the named input; if it persists this is a bug in the planner",),
        )

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        goal = {
            "objective": "domain reasoning assistant",
            "target_capabilities": ["reasoning"],
            "goal": "domain_expert",
            "domain": "manufacturing",
            "preserve_general": True,
            "base_model": {"name_or_path": "some-model", "size_b": 8.0, "arch": "dense"},
            "data": {"sources": [{"uri": "file:///data/corpus", "kind": "local_dir", "approx_tokens": 10_000_000}]},
            "hardware": {"gpu": "H100", "gpus_per_node": 8, "nodes": 1},
            "data_facts": {"has_instructions": True},
        }

        def stage(stage_name: str = "sft", **over: Any) -> dict[str, Any]:
            base = {
                "name": stage_name,
                "stage": stage_name,
                "algorithm": "sft" if stage_name in ("sft", "cpt", "pretrain") else "dr_grpo",
                "method": "lora",
                "recipe_id": "recipe-x",
                "because": "chosen because the fixture needs a valid stage",
                "hparams": {},
                "data": {"format": stage_name, "ready": True, "tokens": 1000},
                "executable": True,
                "missing": None,
                "estimate": {"tokens": 1000, "gpu_hours": 1.0, "hours": 0.125},
                "recipe_provenance": "validated",
                "recipe_derived": False,
                "feasibility": {"verdict": "ok", "findings": [], "alternatives": []},
            }
            base.update(over)
            return base

        def fake_plan(stages: list[dict], plan_goal: dict | None = None) -> dict:
            return {
                "goal": plan_goal if plan_goal is not None else goal,
                "stages": stages,
                "decisions": [],
                "feasibility": {"verdict": "ok", "findings": [], "alternatives": []},
                "provenance_notes": [],
            }

        read = {"verdict": "RED", "dataset_id": "ds-red", "stats": {}, "checks": []}

        # A4.1 (VLM competition LoRA): a VLM goal and a planned VLM stage that
        # satisfy every gate rule except the one a fixture breaks on purpose
        # (vlm_gates in skills/training/planner.py).
        vlm_goal = {
            **goal,
            "modalities": ["image", "text", "video"],
            "video_frames": 8,
            "method": "lora",
            "lora_targets": ["q_proj", "v_proj"],
            "data_facts": {**goal["data_facts"], "modalities": ["image", "text", "video"]},
        }
        vlm_31b_goal = {
            **vlm_goal,
            "base_model": {"name_or_path": "gemma-4-31B-it", "size_b": 31.0, "arch": "dense"},
            "max_sequence_length": 16384,
        }

        def vlm_stage(**over: Any) -> dict[str, Any]:
            base = stage(
                "sft",
                modalities=["image", "text", "video"],
                hparams={"fused_loss": "liger", "lora_targets": ["q_proj", "v_proj"], "max_sequence_length": 16384},
                data={
                    "format": "sft",
                    "ready": True,
                    "tokens": 1000,
                    "fs_columns": {"video_column": "video", "video_frames": 8, "overlong": "drop"},
                },
            )
            base.update(over)
            return base

        # A4.2 (agentic RL): the measured music S0 declaration (examples/agentic_rl/
        # music_s0.json shape: Qwen2.5-1.5B-Instruct, qwen_xml, one vLLM server in the
        # enroot container), one healthy runtime probe, and one violation per TR-IN gate.
        # ``agentic_caps`` is what the harness feeds through the injected probe
        # (TrainingPlannerSkill.__init__): no fixture ever needs a real container.
        agentic = {
            "algorithm": "agentic_grpo_token_mean",
            "policy": {"model_path": "/mnt/models/Qwen2.5-1.5B-Instruct"},
            "trainer": {"max_steps": 3, "lr": 0.0001, "sharding": "none"},
            "engine": {
                "kind": "vllm",
                "weight_sync_mode": "collective_rpc",
                "allow_stale_policy": False,
                "env": {"PATH": "/usr/local/bin:/usr/bin:/bin"},
            },
            "env": {"backend": "local", "allow_unsandboxed": True},
            "harness": {"name": "qwen_xml", "parser": "qwen_xml"},
            "budget": {"step_limit": 4, "max_response_tokens": 256, "max_observation_chars": 1024},
            "sampling": {"temperature": 1.0, "max_new_tokens": 256},
            # handles that exist on any host running the suite (TR-IN-012 checks the real filesystem):
            # the interpreter stands in for an executable abc2midi, this module for a readable tasks file
            "rollout": {"tasks_path": __file__, "tasks_per_step": 1, "group_size": 2},
            "reward": {"kind": "music", "abc2midi_bin": sys.executable},
            "runtime": {"kind": "enroot", "container": "vllmsrv3", "mounts": [["/mnt/models", "/mnt/models"]]},
        }
        agentic_caps = {
            "measured": True,
            "fs_version": "fixture",
            "algorithms": ["agentic_grpo", "agentic_grpo_token_mean"],
            "engines_importable": {"vllm": True},
            "env_backends": ["local"],
            "reward_kinds": ["music"],
            "parsers": ["qwen_xml"],
            "reward_handles": {"abc2midi_bin": sys.executable, "tasks_path": __file__},
            "runtime": {"kind": "enroot", "container": "vllmsrv3"},
            "multi_gpu": None,
            "errors": [],
        }

        def agentic_block(**over: Any) -> dict[str, Any]:
            built = {**agentic}
            built.update(over)
            return built

        def agentic_goal(block: dict[str, Any], **over: Any) -> dict[str, Any]:
            built = {**goal, "agentic": block, "hardware": {"gpu": "GB200", "gpus_per_node": 1, "nodes": 1}}
            built.update(over)
            return built

        agentic_stage = stage(
            "agentic_rl",
            name="agentic0",
            algorithm="agentic_grpo_token_mean",
            method="full",
            hparams={"max_steps": 3},
        )
        fixtures: dict[str, dict[str, Any]] = {
            "TR-IN-001": {"request": {"goal": {**goal, "base_model": {}}}},
            "TR-IN-002": {"request": {"goal": {**goal, "objective": ""}}},
            "TR-IN-003": {"request": {"goal": {**goal, "hardware": {}}}},
            "TR-IN-004": {"request": {"goal": goal, "readiness": read}},
            "TR-PL-001": {
                "request": {"goal": goal},
                "fake_plan": fake_plan([stage("sft", executable=False, missing="missing: sft objective")]),
            },
            "TR-PL-002": {
                "request": {"goal": goal},
                "fake_plan": fake_plan(
                    [
                        stage(
                            "sft",
                            feasibility={
                                "verdict": "infeasible",
                                "findings": [{"symptom": "oom"}],
                                "alternatives": [{"change": "lora", "verdict": "infeasible", "executable": False}],
                            },
                        )
                    ]
                ),
            },
            "TR-PL-003": {
                "request": {"goal": goal},
                "fake_plan": fake_plan([stage("sft", recipe_provenance="literature")]),
            },
            "TR-PL-004": {
                "request": {"goal": goal},
                "fake_plan": fake_plan(
                    [stage("sft", data={"format": "sft", "ready": False, "tokens": 1000, "handoff": "data_engine", "target_format": "sft"})]
                ),
            },
            "TR-PL-005": {"request": {"goal": goal}, "fake_plan": fake_plan([stage("sft")])},
            # A4.1: input rules fire from check_inputs; TR-PL-006/007 (RED) fire from
            # check_handoff over the planned stages. TR-PL-008 is a request fixture:
            # it is an input-phase rule (see its RuleSpec description).
            "TR-IN-005": {"request": {"goal": {**vlm_goal, "lora_targets": None}}},
            "TR-IN-006": {"request": {"goal": {**vlm_goal, "video_frames": None}}},
            "TR-IN-007": {"request": {"goal": {**vlm_goal, "lora_targets": ["  ", ""]}}},
            "TR-IN-008": {"request": {"goal": {**vlm_goal, "tp": 2, "cp": 2}}},
            "TR-PL-006": {
                "request": {"goal": {**vlm_goal, "max_sequence_length": 16384}},
                "fake_plan": fake_plan(
                    [
                        vlm_stage(
                            hparams={
                                "fused_loss": "liger",
                                "lora_targets": ["q_proj", "v_proj"],
                                "max_sequence_length": 131072,
                            }
                        )
                    ],
                    plan_goal={**vlm_goal, "max_sequence_length": 16384},
                ),
            },
            "TR-PL-007": {
                "request": {"goal": vlm_31b_goal},
                "fake_plan": fake_plan(
                    [
                        vlm_stage(
                            hparams={
                                "fused_loss": "liger",
                                "lora_targets": ["q_proj", "v_proj"],
                                "max_sequence_length": 65536,
                            }
                        )
                    ],
                    plan_goal=vlm_31b_goal,
                ),
            },
            "TR-PL-008": {"request": {"goal": {**vlm_goal, "max_sequence_length": 32768}}},
            # A4.2 (agentic): one violation per new rule; ``agentic_caps`` is the runtime
            # probe the harness feeds through the injected ``agentic_probe``.
            "TR-IN-010": {
                "request": {"goal": agentic_goal(agentic_block(harness={"name": "qwen_xml", "parser": "regex_v1"})), "agentic_probe": agentic_caps},
                "agentic_caps": agentic_caps,
            },
            "TR-IN-011": {
                "request": {
                    "goal": agentic_goal(agentic_block(trainer={"max_steps": 3, "lr": 1e-6, "sharding": "fsdp"})),
                },
                "agentic_caps": agentic_caps,
            },
            "TR-IN-012": {
                "request": {
                    "goal": agentic_goal(agentic_block(reward={"kind": "music", "abc2midi_bin": "/usr/bin/abc2midi"})),
                },
                # the probe measured the handle the music scorer reads and it is NOT executable there
                "agentic_caps": {**agentic_caps, "reward_handles": {"abc2midi_bin": None}},
            },
            "TR-IN-013": {
                "request": {
                    "goal": agentic_goal(
                        agentic_block(
                            trainer={"max_steps": 3, "lr": 1e-6, "sharding": "none"},
                            engine={"kind": "vllm", "weight_sync_mode": None, "allow_stale_policy": False,
                                    "env": {"PATH": "/usr/local/bin:/usr/bin:/bin"}},
                        )
                    ),
                },
                "agentic_caps": agentic_caps,
            },
            "TR-IN-014": {
                "request": {
                    "goal": agentic_goal(
                        agentic_block(engine={"kind": "vllm", "weight_sync_mode": "collective_rpc", "env": {}}),
                    ),
                },
                "agentic_caps": agentic_caps,
            },
            "TR-PL-009": {
                "request": {"goal": agentic_goal(agentic_block()), "agentic_probe": agentic_caps},
                "fake_plan": fake_plan([agentic_stage], plan_goal=agentic_goal(agentic_block())),
                "agentic_caps": agentic_caps,
            },
        }
        # A4.2: the generic fixture harness calls execute(request) only, so each agentic fixture's runtime
        # probe travels in the request as a stored, measured probe (no container is started in tests)
        for fixture in fixtures.values():
            if "agentic_caps" in fixture:
                fixture["request"].setdefault("agentic_probe", fixture["agentic_caps"])
        return fixtures

