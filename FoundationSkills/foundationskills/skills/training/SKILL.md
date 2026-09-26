
# training.planner — FoundationSkills training planning

## Purpose

Turn a high-level `goal_spec` (optionally with a data `readiness_report`) into a
`training_plan`: an ordered set of stages (pretrain/cpt/sft/preference/rl) where
each stage either is **executable on the installed FoundationScale** or names
the exact missing capability. Every decision is recorded in the plan's
`decisions` log so the plan can be audited before anything is launched.

## When to use

- A user/agent states a training goal ("a manufacturing-domain reasoning model
  from a 7B base") and you need to decide stages, algorithms, method (full /
  LoRA / QLoRA), hyperparameters, and feasibility — against *measured* FS
  capabilities, never assumptions.
- Before calling `training.emit` (which renders the plan into an
  `fs_launch_spec`).
- To diagnose a training failure via `diagnose` / `diagnose_symptom`.

Do **not** use this skill to prepare data (that is `data_engine`) or to run
evals. A RED readiness report is refused: fix the data first.

## Inputs

Request object:

| field | type | required | meaning |
|---|---|---|---|
| `goal` | `goal_spec` payload | yes | objective, base_model, data sources, hardware, preserve_general, domain |
| `readiness` | `readiness_report` payload | no | measured data readiness; a PASS marks stage data as ready |
| `hardware_id` | string | no | knowledge hardware id overriding `goal.hardware.gpu` fuzzy mapping |

Consumes artifact types: `goal_spec` (readiness optional).

## Outputs

Produces artifact type `training_plan`, written to `<workdir>/artifacts/` and
returned in the result payload under `plan`. Per stage: name, stage, algorithm,
method, recipe_id, because (never empty), hparams, data (with a
`{"handoff": "data_engine", "target_format": ...}` when not ready), executable,
missing, estimate (FLOPs/tokens-per-second/GPU-hours/memory), and a per-stage
feasibility verdict with alternatives. Plan level: overall feasibility,
decisions for all ten planning steps (goal, model_analysis, data_analysis,
hardware_analysis, stage_selection, algorithm_selection, recipe_match, method,
estimate, feasibility), provenance_notes, and open_questions (missing
load-bearing facts from `foundationskills.agent.intake`).

## Scope

- model_types: llm, vlm; families: any (unregistered families trigger the
  `--adapter-target` LoRA decision).
- stages: pretrain, cpt, sft, preference, rl; methods: full, lora, qlora
  (QLoRA is not executable on the installed FS — alternatives mark it
  `executable: false`).
- hardware: gb200, h100, a100 knowledge entries (fuzzy GPU-name mapping).
- FS facts this skill plans around: only `--objective sft` exists (CPT/pretrain
  run the same entry over a `text` corpus); SFT trains on the *full rendered
  text* (no assistant-only masking — disclosed via TR-PL-005); DDP/FSDP only,
  tp>1/cp>1 execute, pp>1/ep>1 are refused; RL is library-only, single-device,
  with only dr_grpo/gspo/dapo runnable.

## Validation rules

| rule id | severity | phase | fires when |
|---|---|---|---|
| TR-IN-001 | BLOCK | input | `goal.base_model.name_or_path` missing/empty |
| TR-IN-002 | BLOCK | input | `goal.objective` missing/empty |
| TR-IN-003 | BLOCK | input | `goal.hardware` missing gpu/gpus_per_node/nodes |
| TR-IN-004 | BLOCK | input | a RED readiness_report supplied — fix the data first; hand off to data_engine |
| TR-PL-001 | WARN | handoff | plan has 0 executable stages (still useful as a gap list) |
| TR-PL-002 | BLOCK | handoff | a stage is infeasible with no executable feasible alternative |
| TR-PL-003 | INFO | handoff | a stage relies on an unvalidated (literature/community/derived) recipe |
| TR-PL-004 | WARN | handoff | a stage needs data not yet prepared (handoff data_engine) |
| TR-PL-005 | INFO | handoff | an sft stage is present: FS SFT full-sequence loss disclosure |
| FS-IN-001 | BLOCK | input | (training.emit) the plan has no stages |
| FS-IN-002 | BLOCK | input | (training.emit) a stage has no dataset payload |
| FS-IN-003 | BLOCK | input | (training.emit) FoundationScale is not importable: refusal names it |
| FS-HO-001 | WARN | handoff | (training.emit) a launch spec is non-executable; `missing` names the gap (e.g. LoRA adapter needs a merge before the next stage; FS RL saves no checkpoint) |
| FS-HO-002 | BLOCK | handoff | (training.emit) every launch spec is non-executable |
| FS-HO-003 | INFO | handoff | (training.emit) an RL spec is single-device; runnable as measurement-only (`fskills launch --measurement-only`) |

Statuses: PASS 0 / RED 5 / UNMEASURED 95 / REFUSED 96. A `PlanningRefusal`
(no stage selected; unknown model with no size; unknown hardware) is REFUSED
and names the missing input.

## Failure handling

`diagnose` accepts a `SkillResult`, an exception (`PlanningRefusal`,
`KnowledgeError`, or anything raised at training time), or a symptom string.
Playbook keys:

| symptom | gist |
|---|---|
| `loss_spike` | bad shard / LR too high / bf16 overflow — resume pre-spike, halve LR or drop the shard |
| `divergence_nan` | NaN loss — lower LR, gradient clipping, validate records |
| `oom` | batch/seq too big, optimizer not sharded — micro-batch 1 + accumulation, full→lora, halve seq (never pp/ep: refused) |
| `reward_hacking` | exploitable verifier — harden answer_pattern, raise group_size |
| `reward_saturation_unmeasured_steps` | FS prints "UNMEASURED step N" when all advantages are zero — raise group_size/temperature, use harder prompts |
| `entropy_collapse` | RL LR too high / sampling too narrow |
| `moe_router_imbalance` | aux loss too small / over-specialised routing |
| `slow_throughput` | dataloader starvation / over-estimated MFU / wrong sharding |
| `fs_refused_96` | exit 96: read the `[fs:train:refuse]` line; it names the missing input |
| `hang_no_progress` | job RUNNING but the log stalled: judge by GPU util + log mtime, never scheduler state; look for a dead rank or DataLoader worker |
| `lr_not_applied` | logged LR differs from the plan: check warmup arithmetic and that the scheduler state was resumed |

## FS interface

- entries: `foundationscale-train` (supervised stages; `--dry-run` validates
  with 0 GPUs), `fskills-rl` (RL driver around
  `foundationscale.rl.trainer.RLTrainer(RLTrainConfig).run()`).
- emits: `training_plan`; the sibling skill `training.emit` renders
  `fs_launch_spec` payloads (+ sbatch with the GB200 IMEX-fabric preamble and
  `--time=10-00:00:00 --exclude=r01gb200...` cluster rules).
- Runnability always comes from `FSCapabilities.check` / `rl_runnable` —
  registered is not runnable.

## Worked examples

### 1. Manufacturing reasoning model from an open 7B (the reference scenario)

Goal: Llama-3.1-8B base, ~2B tokens of internal documents + 20k QA pairs with
verifiable answers, 2 nodes x 8 H100, preserve general capabilities.
Plan: `cpt` (full, size-band LR 2e-5, token budget min(4 epochs, 20x domain
capped), replay ~0.25 from the mixing rules) -> `sft` (LoRA; <50M tokens) ->
`rl` on 1 GPU (`dr_grpo`: "GRPO requested; FS runs dr_grpo (reference-free)
today"). The llama3 family is not FS-registered, so the plan records the
`--adapter-target` decision; sft/rl data carry data_engine handoffs.

### 2. Math RL on gemma4 E4B (GB200)

Goal: math reasoning, E4B base with verifiable answers. Stage rules keep a
single `rl` stage (base already instruct): algorithm `dr_grpo` (or
`gspo`/`dapo` per `caps.rl_runnable`), method full-or-lora whichever fits
189 GB on one GB200, `rl_generation_factor` 3.0 in the time estimate. If the
corpus request is too easy, expect `reward_saturation_unmeasured_steps`.

### 3. General-chat SFT LoRA on qwen3.5-27B

Goal: general_chat from the 27B instruct base. One `sft` stage, method `lora`
(CPVT-size data is far below the 50M-token full-tuning threshold), LoRA targets
from the family card's `lora_targets`. TR-PL-005 discloses full-sequence loss;
the recipe provenance is flagged TR-PL-003 when the match is literature.

## Sub-skills

- `planner.py` — the ten-step plan builder (`plan`, `PlanningRefusal`).
- `cpt.py` — `cpt_policy`: size-band LR, token budget, replay ratio.
- `posttrain.py` — `select_posttrain_stages`, `select_algorithm` (RL/preference
  selection against `caps.rl_runnable`).
- `method.py` / `estimate.py` / `feasibility.py` (D1) — method choice,
  memory/time models, feasibility with alternatives.
- `recipes.py` / `knowledge.py` (D1) — recipe matching and the knowledge base
  (`families/`, `hardware/`, `algorithms/`, `recipes/`, `stage_rules.yaml`).
- `emit_skill.py` (E) — `training.emit`, rendering plans to `fs_launch_spec`.
