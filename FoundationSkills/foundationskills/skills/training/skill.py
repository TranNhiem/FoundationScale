
"""TrainingPlannerSkill: goal_spec (+optional readiness) -> training_plan artifact."""
from __future__ import annotations

import re
from typing import Any

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
from foundationskills.skills.training.planner import PlanningRefusal, plan

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


class TrainingPlannerSkill(BaseSkill):
    """Plan an executable-or-explicitly-refused multi-stage training run."""

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
        RuleSpec("TR-PL-001", "plan has 0 executable stages on the installed FS", Severity.WARN, "handoff"),
        RuleSpec("TR-PL-002", "a stage is feasibility-infeasible with no executable feasible alternative", Severity.BLOCK, "handoff"),
        RuleSpec("TR-PL-003", "a stage relies on an unvalidated (literature/community/derived) recipe", Severity.INFO, "handoff"),
        RuleSpec("TR-PL-004", "a stage needs data not yet prepared (handoff to data_engine)", Severity.WARN, "handoff"),
        RuleSpec("TR-PL-005", "disclosure: FS SFT trains on the full rendered text (no assistant-only loss masking)", Severity.INFO, "handoff"),
    )
    fs_interface = FSInterface(
        entries=("foundationscale-train", "fskills-rl"),
        emits=("training_plan",),
        apis=("foundationscale.train.cli", "foundationscale.rl.trainer.RLTrainer"),
        notes="This skill only plans; training.emit turns the plan into an fs_launch_spec. --dry-run runs the FS validation prologue with 0 GPUs.",
    )

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

        def fake_plan(stages: list[dict]) -> dict:
            return {
                "goal": goal,
                "stages": stages,
                "decisions": [],
                "feasibility": {"verdict": "ok", "findings": [], "alternatives": []},
                "provenance_notes": [],
            }

        read = {"verdict": "RED", "dataset_id": "ds-red", "stats": {}, "checks": []}
        return {
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
        }
