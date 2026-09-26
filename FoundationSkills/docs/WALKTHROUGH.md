# Walkthrough: the reference scenario

> "I want to build a manufacturing-domain reasoning model from an open-source 7B base, using our internal documents, on our DGX H100 cluster, while preserving general capabilities."

Every output below was produced by the real skills against the installed FoundationScale, on 2026-09-26. Regenerate it with the script in `tests/scenario/` or the commands shown. Numbers are estimates, and each one says where its assumptions come from.

## 1. Understand the goal

The agent turns the request into a `goal_spec`. `agent/intake.missing_facts()` produces the questions it must ask: base model, where the data is and how big it is, target capability, hardware and budget. The answers used here:

| Fact | Value |
|---|---|
| Base model | `meta-llama/Llama-3.1-8B`, which resolves to the knowledge-base variant `llama-3.1-8b` (8.03B, dense) |
| Internal documents | 2B tokens of manufacturing documents, tagged `use_for: [cpt]` |
| Domain Q&A | 40k examples / 30M tokens, `use_for: [sft]` |
| Verifiable data | 8k process-exam multiple-choice questions, `use_for: [rl]`, `answer_kind: mcq_letter` |
| Hardware | DGX H100, 2 nodes × 8 GPUs |
| Budget | 3,000 GPU-hours |
| Constraint | preserve general capability |

## 2. Probe what FoundationScale actually runs

`fskills probe --deep` reports:
- the `sft` objective (causal LM over `text`);
- DDP and FSDP;
- tp/cp execute, pp/ep are refused;
- 18 registered RL algorithms, of which **dr_grpo, gspo and dapo run**;
- RL rewards `mcq_letter` only;
- RL saves **no** checkpoint;
- families `gemma4` and `qwen3.5` (Llama is not FS-registered, so LoRA needs explicit `--adapter-target`).

## 3. The plan (`training.planner`)

| Stage | Algorithm | Method | Recipe | Estimate (16×H100 unless noted) | Executable |
|---|---|---|---|---|---|
| CPT | causal LM | full | `llama3-8b-manufacturing-cpt` | 8.0B tokens, 25.5 GB/GPU, **270.6 GPU-h** | yes |
| SFT | sft | LoRA r32 | `llama3-8b-manufacturing-sft-lora` | 30M tokens × 2 epochs, 33.0 GB/GPU, 0.7 GPU-h | yes |
| RL | dr_grpo | LoRA | `llama3-8b-reasoning-rl` | 81.9M trained tokens, **1 GPU**, 35.2 GB, 5.5 GPU-h | **no, measurement-only** |

Feasibility: **ok**, about 277 GPU-hours against a budget of 3,000.

Why each decision was made (quoted from the plan's `decisions`):
- **Stages cpt → sft → rl.** "≥100M domain tokens is enough for continued pretraining to move the base model's distribution; SFT alone cannot absorb that volume." "The base model has no instruction tuning; an SFT stage is mandatory before RL." "Verifiable gold answers enable reference-free group RL." The order is the canonical lifecycle.
- **CPT: full fine-tune.** "Full fine-tuning is required: low-rank LoRA adapters lack the capacity to absorb a corpus-scale distribution shift."
  - LR 2e-5 with rewarm and cosine re-decay (Ibrahim et al. 2024).
  - **30% general replay** to preserve general ability. `data_engine.mix` designs the mixture.
  - Token budget 8B: 2B domain tokens with an epoch cap of 4.
- **SFT: LoRA.** "SFT corpus is small (30M tokens < 50M); LoRA is cheaper and forgets less of the base model." The LoRA recipe was chosen *because* LoRA was chosen, since recipe hparams are method-specific.
- **RL: Dr.GRPO.** "GRPO requested; FS runs dr_grpo (reference-free) today. FS refuses grpo." LoRA because "FS RLTrainer is single-device and full RL needs ~150 GB > 64 GB on one GPU".
- **RL is measurement-only.** "dr_grpo runs on the installed FS, but FS RLTrainer persists no trained policy, so the stage can be run to measure rewards and throughput and cannot hand weights to a later stage (core gap)."
- **Provenance.** All three recipes are `literature` (not yet validated on FoundationScale), and the CPT and SFT matches are `derived`: the recipes target `domain_expert` while the goal is `reasoning`. The plan says so.

## 4. The data (`data_engine`)

The planner hands each stage's format to the Data Engine, which recommends a pipeline:

| Target | Pipeline | Key decisions |
|---|---|---|
| CPT corpus | ingest (documents) → clean (PII, HTML) → dedup (exact + MinHash) → quality (Gopher + FineWeb) → [decontam] → mix (domain 0.7 / general replay 0.3) → format `{text}` → tokenize (chunk, pack) | FS truncates at `max_sequence_length`, so long documents are **chunked** at paragraph boundaries rather than truncated |
| SFT | ingest → clean → dedup (per example) → quality (SFT checks) → decontam → format `{messages, text}` using Llama-3's chat template → tokenize (`drop_overlong` available) | full-sequence loss is **disclosed**, because FS has no assistant-only masking |
| RL | ingest (multiple-choice) → … → format: question + "A. … B. …" + "Answer with a single letter.", gold letter | FS rewards only single-letter gold; free-form answers are dropped with a named reason |

Training starts only when a stage's readiness verdict is **PASS**. Measured on real corpora in `docs/VALIDATION.md`.

## 5. Ready-to-run FoundationScale commands (`training.emit`, hardware `h100-80gb`, 2×8)

**CPT** (sbatch with `--time=10-00:00:00`; `cd` to the FoundationScale repo root so the run records its commit):
```
torchrun --nnodes 2 --nproc-per-node 8 --rdzv-backend c10d --rdzv-endpoint ${MASTER_ADDR}:29500 \
  -m foundationscale.train.cli --model meta-llama/Llama-3.1-8B --dataset /data/mfg/prepared/cpt_mix/shards \
  --output-dir /checkpoints/mfg-reasoning/mfg-cpt --nodes 2 --gpus-per-node 8 --profile-name slurm-generic --dp 16 \
  --objective sft --max-steps 477 --learning-rate 2e-05 --lr-scheduler-type cosine --max-sequence-length 4096 \
  --per-device-batch-size 2 --gradient-accumulation-steps 128 --precision bf16 --sharding-strategy fsdp \
  --save-interval 1000 --seed 0 --optimizer adamw_torch_fused --logging-steps 10 --max-grad-norm 1.0 \
  --warmup-steps 200 --gradient-checkpointing true
```
Spec notes: `min_lr_ratio` (the policy's 10% floor) has no FS flag and is reported, not passed. FS's cosine decays to zero.

**SFT**, starting from `mfg-cpt/final` (stage chaining):
```
... --model /checkpoints/mfg-reasoning/mfg-cpt/final --dataset /data/mfg/prepared/sft/shards ... --max-steps 29
  --learning-rate 0.0001 --per-device-batch-size 4 --gradient-accumulation-steps 8 ... --warmup-steps 3
  --adapter lora --adapter-rank 32 --adapter-alpha 64.0 --adapter-dropout 0.05
  --adapter-target q_proj --adapter-target k_proj --adapter-target v_proj --adapter-target o_proj
  --adapter-target gate_proj --adapter-target up_proj --adapter-target down_proj
```
Warmup was clamped from the recipe's 100 to 10% of the run's 29 steps, and the spec notes say so.

**RL**, `executable: false`. The spec names both blocking gaps:
1. `FS RLTrainer does not persist the trained policy (no checkpoint)`;
2. `merge the 'sft' LoRA adapter into the base model (FS writes adapter-only checkpoints; future merging skill)`.

The `rl_config` is ready (`dr_grpo`, group 8, lr 1e-6, gold key `answer`). An operator may run it with `fskills launch --measurement-only` to measure rewards and throughput.

## 6. Confirm and launch

```
fskills hash   --spec emit/launch/fs_launch_spec.mfg-cpt.json      # show this hash to the user
fskills launch --spec emit/launch/fs_launch_spec.mfg-cpt.json --confirm <hash-from-user>
```
Without the correct hash, the launch refuses with exit 96: "ask the user; never auto-fill". The launcher first runs FS's `--dry-run`, then submits. It keeps the full log. When torchrun reports exit 1, it recovers FS's own verdict (0/5/95/96) from FS's log line.

## 7. Monitor and next step

- **Monitoring.** Training symptoms map to diagnoses via `TrainingPlannerSkill.diagnose_symptom(...)`: `loss_spike`, `divergence_nan`, `oom`, `reward_hacking`, `reward_saturation_unmeasured_steps`, `entropy_collapse`, `moe_router_imbalance`, `slow_throughput`, `fs_refused_96`.
- **What this plan delivers today.** A CPT'd and SFT'd Llama-3.1-8B (the SFT output is a LoRA adapter on the CPT weights).
- **What it needs for RL.** The merging skill (future), plus two FoundationScale core features: RL checkpoint persistence, and multi-GPU RL for scale.
- **After evaluation.** An `eval_report`, from the future benchmarking skill, goes back to the planner to recommend the next iteration.
