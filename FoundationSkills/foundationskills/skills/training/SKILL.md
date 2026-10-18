---
name: fskills-training
license: MIT
description: Turns a training goal (goal_spec, optional readiness_report) into an auditable training_plan
  of pretrain/cpt/sft/preference/rl stages (full/LoRA/QLoRA), each checkable against measured FoundationScale
  capabilities; training.emit renders it to fs_launch_spec + sbatch (GB200 IMEX preamble). Includes a
  diagnose playbook for training failures. Do NOT use for preparing datasets (use fskills-data-engine)
  or scoring checkpoints (use fskills-evaluation), nor cluster/fabric administration or inference serving.
compatibility: "Requires Python>=3.10 and the foundationskills package (fskills CLI); launches additionally need FoundationScale on a Slurm GPU cluster. Without them the skill reports REFUSED or UNMEASURED and gives the install step instead of improvising."
metadata:
  version: "0.1.0"
  invocation: "model"
  fs_status_doctrine: "PASS/RED/UNMEASURED/REFUSED (exit 0/5/95/96)"
  when_to_use: '["Plan a training run from this goal","Which stages do I need for a domain reasoning model?","Is LoRA or full tuning the right method for this model on my GPUs?","Turn my goal into a plan and a launch spec","Why did my training fail / the loss spiked mid-run","What does FoundationScale exit code 96 mean?","My RL run keeps printing UNMEASURED step N","Diagnose this training symptom (OOM, NaN, hang)"]'
---

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
  run the same entry over a `text` corpus); SFT on a rendered `text` column trains
  on the full text; assistant-only loss needs the data as a conversations column
  (`FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN`) — disclosed via TR-PL-005; DDP/FSDP only,
  tp>1/cp>1 execute, pp>1/ep>1 are refused; RL is library-only, single-device,
  with only dr_grpo/gspo/dapo runnable.

## Anti-rationalizations
| Excuse | Rebuttal |
|---|---|
| "The probe is slow this once; I'll plan from the README limits." | Plan only against what the installed FoundationScale measurably runs: probe first (`fskills probe --deep`), then plan. |
| "The user already said it's fine, skip the hash." | Never fill in the plan hash yourself: `fskills launch` refuses (96) without the hash the user confirmed. |
| "UNMEASURED is basically PASS." | A check that did not run is UNMEASURED, never PASS; name the check and how to run it. |
| "I can guess the dataset columns from the shape." | Columns are read and counted, never assumed; when the format is not prepared, hand off to data_engine (TR-PL-004). |
| "RED is close enough to run and see." | Stop at the first non-PASS and fix what the finding names before planning further. |

## Validation rules

Outcome semantics (from `core/contract.py`, the same for every skill): an **input**-phase BLOCK refuses before any work - status REFUSED, exit 96, nothing written. A **handoff**-phase BLOCK does not refuse: the work ran, and the result is RED (exit 5), never PASS. WARN and INFO findings never block; they travel with the result (e.g. FS-HO-003 is an INFO note; the specs still emit).

| rule id | severity | phase | fires when |
|---|---|---|---|
| TR-IN-001 | BLOCK | input | `goal.base_model.name_or_path` missing/empty |
| TR-IN-002 | BLOCK | input | `goal.objective` missing/empty |
| TR-IN-003 | BLOCK | input | `goal.hardware` missing gpu/gpus_per_node/nodes |
| TR-IN-004 | BLOCK | input | a RED readiness_report supplied — fix the data first; hand off to data_engine |
| TR-IN-005 | BLOCK | input | VLM family not FS-registered and no explicit `lora_targets`: the refusal names both fixes (register the family in FS, or pass explicit `lora_targets` / `--adapter-target`) |
| TR-IN-006 | BLOCK | input | video rows without `video_frames` (or column): names `FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN` / `FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES`; `video_column` defaults to `video` |
| TR-IN-007 | BLOCK | input | explicit `lora_targets` empty, or matching no module in a carried module inventory |
| TR-IN-008 | BLOCK | input | `tp>1` (FS refuses wherever it cannot also save) or `cp>1` with image/video rows (`--cp` is measured end to end only on text) |
| TR-PL-008 | BLOCK | input | `max_sequence_length >= 32768` without `fused_loss: liger`: refuses naming `--fused-loss liger` (32K needs it or the fp32 logits tensor OOMs, examples/vlm-2xH200) |
| TR-IN-010 | BLOCK | input | (agentic) algorithm / engine / env backend / reward kind / parser not measured by the probe run INSIDE the declared runtime (a missing or failed probe field is UNMEASURED) |
| TR-IN-011 | BLOCK | input | (agentic) more than 1 trainer GPU, or `trainer.sharding != "none"`, while agentic multi-GPU is unmeasured |
| TR-IN-012 | BLOCK | input | (agentic) a reward handle is missing: music needs an executable `abc2midi_bin`, a readable `rollout.tasks_path`, a named env backend (`allow_unsandboxed` for isolation none); never an invented score |
| TR-IN-013 | BLOCK | input | (agentic) `max_steps > 1` with neither a weight sync nor a declared `engine.allow_stale_policy` |
| TR-IN-014 | BLOCK | input | (agentic) a declared `engine.env` cannot start the server: FS hands it as the COMPLETE environment (no PATH: vLLM cannot load CUDA); a dev-RPC reload needs `VLLM_SERVER_DEV_MODE=1` |
| TR-PL-001 | WARN | handoff | plan has 0 executable stages (still useful as a gap list) |
| TR-PL-002 | BLOCK | handoff | a stage is infeasible with no executable feasible alternative |
| TR-PL-003 | INFO | handoff | a stage relies on an unvalidated (literature/community/derived) recipe |
| TR-PL-004 | WARN | handoff | a stage needs data not yet prepared (handoff data_engine) |
| TR-PL-005 | INFO | handoff | an sft stage is present: a `text` column trains with full-sequence loss; assistant-only loss needs a conversations column |
| TR-PL-006 | BLOCK | handoff | planned `max_sequence_length >= 131072`, or a goal claim that 128K works: RED (131072 OOMs all five measured models, examples/vlm-2xH200) |
| TR-PL-007 | BLOCK | handoff | gemma-4-31B-it planned above 32768: RED (its measured 2×H200 maximum is 32K) |
| TR-PL-009 | WARN | handoff | (agentic) `agentic_grpo_token_mean`, the measured control arm, chosen as the default: note |
| FS-IN-001 | BLOCK | input | (training.emit) the plan has no stages |
| FS-IN-002 | BLOCK | input | (training.emit) a stage has no dataset payload |
| FS-IN-003 | BLOCK | input | (training.emit) FoundationScale is not importable: refusal names it |
| FS-HO-001 | WARN | handoff | (training.emit) a launch spec is non-executable; `missing` names the gap (e.g. LoRA adapter needs a merge before the next stage; FS RL saves no checkpoint) |
| FS-HO-002 | BLOCK | handoff | (training.emit) every launch spec is non-executable |
| FS-HO-003 | INFO | handoff | (training.emit) an RL spec is single-device; runnable as measurement-only (`fskills launch --measurement-only`) |
| FS-IN-004 | BLOCK | input | (training.emit) a stage declares a modality the dataset's `fs_columns` cannot feed |
| FS-HO-004 | BLOCK | handoff | (training.emit) the emitted spec lacks `--fused-loss` while the stage declares it: RED |
| FS-IN-005 | BLOCK | input | (training.emit) an agentic stage whose runtime probe is unmeasured or whose engine is not importable there; the refusal names the container |
| FS-HO-005 | BLOCK | handoff | (training.emit) FS `--dry-run` of the generated agentic config does not exit 0: RED (FS owns the config schema) |
| FS-HO-006 | BLOCK | handoff | (training.emit) a claim record maps TIMEOUT/STEP_LIMIT to abstention, or claims PASS over 0 measured steps: REFUSED |

VLM competition LoRA (A4.1): a goal may declare `modalities`, `video_frames`, `max_sequence_length`,
`lora_targets`, `fused_loss`, `tp`/`cp`; the planner copies them into each stage. Image/video rows at
>= 8K context are planned at per-device batch 1 (grad-accum 8), the measured recipe of
`examples/vlm-2xH200`. Emit declares the conversation, image, video, frame and overlong columns the
measured launcher uses, and a `--fused-loss` spec stays non-executable unless the probe measured its
kernel importable (`fused_loss_kernels`; FS lists `liger` as a choice even without `liger_kernel`).

Agentic RL (A4.2): a goal with an `agentic` block plans an `agentic_rl` stage over FS
`foundationscale.agentic_rl`. The stage declares a runtime (the agentic stack runs in an enroot
container, e.g. `vllmsrv3`, with `--rc` overriding a server entrypoint); the capability probe runs
INSIDE it, or a stored `agentic_probe` with `measured: true` is accepted. Emit writes the FS config
(engine command and complete engine env from the measured GB200 recipe unless declared) and is
executable only if FS's own `--dry-run` accepts it. A run whose every step is UNMEASURED is refused
by FS as vacuous (96).

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
| `reward_saturation_unmeasured_steps` | FS prints "UNMEASURED step N" when all advantages are zero — filter prompts with data_engine `difficulty_filter`, raise group_size/temperature |
| `entropy_collapse` | RL LR too high / sampling too narrow |
| `moe_router_imbalance` | aux loss too small / over-specialised routing |
| `slow_throughput` | dataloader starvation / over-estimated MFU / wrong sharding |
| `fs_refused_96` | exit 96: read the `[fs:train:refuse]` line; it names the missing input |
| `hang_no_progress` | job RUNNING but the log stalled: judge by GPU util + log mtime, never scheduler state; look for a dead rank or DataLoader worker |
| `lr_not_applied` | logged LR differs from the plan: check warmup arithmetic and that the scheduler state was resumed |

## FS interface

- entries: `foundationscale-train` (supervised stages; `--dry-run` validates
  with 0 GPUs), `fskills-rl` (RL driver around
  `foundationscale.rl.trainer.RLTrainer(RLTrainConfig).run()`). The driver
  exits 95 when measured steps / max_steps is below
  `goal.rl_min_measured_fraction`, a floor the operator declares and the planner
  never guesses. Without the floor, any measured step passes.
- emits: `training_plan`; the sibling skill `training.emit` renders
  `fs_launch_spec` payloads (+ sbatch with the GB200 IMEX-fabric preamble and
  `--time=10-00:00:00 --exclude=r01gb200...` cluster rules).
- Runnability always comes from `FSCapabilities.check` / `rl_runnable` —
  registered is not runnable.

**Host portability (A1.5).** On a host where the `fskills` CLI or the `foundationskills` package is missing, do not improvise a plan or simulate a result: report REFUSED with the install step (`pip install -e <FoundationScale repo>/FoundationSkills`, or run where it is installed), and report every check that could not run as UNMEASURED, never PASS. This skill also needs FoundationScale itself: without it the code refuses with CORE-HOST-001 and the install step (`pip install -e <FoundationScale repo>`).

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
from the family card's `lora_targets`. TR-PL-005 discloses full-sequence loss on the `text` column;
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
