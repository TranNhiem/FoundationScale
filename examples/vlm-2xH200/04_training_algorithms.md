# 04 — Training algorithms: SFT, DPO-family preference training and GRPO-family RL

FoundationScale registers 18 algorithms behind three front doors. Everything you learned in chapters 02–03 still applies (data loading, LoRA, bf16, acceptance runs); this chapter shows how to run preference training and RL on your 2× H200 node and what the measured numbers look like.

Every command below assumes the environment is active — activate it first in a fresh shell.

| Front door | Trainer / CLI | `algorithm=` values | Modality support |
|---|---|---|---|
| SFT | `foundationscale-train` CLI (chapter 03) | `sft` | image + video + text |
| Offline preference | `PreferenceTrainer` | `dpo`, `ipo`, `kto`, `orpo`, `simpo`, `cpo` | **image + video + text** — declare `image_column`/`video_column` (`--image-column`/`--video-column`) to train on them; a row carrying media in an UNDECLARED column is refused (exit 96) |
| Online RL | `RLTrainer` | `grpo`, `dr_grpo`, `dapo`, `gspo`, `reinforce_baseline`, `reinforce_pp`, `rloo`, `raft`, `best_of_n`, `online_dpo`, `iterative_dpo`, `ppo` | **images yes, video refused** |

Switching algorithm is a one-line change: the `algorithm` field of `PreferenceTrainConfig` / `RLTrainConfig` (command-line: `--algorithm`; the `objective` name appears in some older configs, but the shipped scripts use `algorithm`). Everything else — data format, sharding, LoRA fields — stays the same.

Two runbook properties you get for free on every algorithm:

- **LoRA field names mirror SFT**: `adapter="lora"`, `adapter_rank`, `adapter_alpha`, `adapter_dropout`, `adapter_targets`. The reference model is *the same model with the adapter switched off*, so no second copy is ever loaded.
- **Sanity checks that must hold exactly**: at step 0 the DPO loss is exactly `ln 2 = 0.6931` and the GRPO KL is exactly `0`. If they don't, stop and see Troubleshooting.

---

## Step 1 — Prepare the two datasets

`prepare_rl_data.py` emits both formats. Run from the repository root `FoundationScale/` (the parent of `models/`); every command in this chapter runs from there unless stated.

1. Generate the text-only preference set (`{"prompt","chosen","rejected"}` JSONL rows):

```bash
python3 examples/vlm-2xH200/scripts/prepare_rl_data.py preference \
  --out data/pref/dpo_train.jsonl --rows 2000
```

2. Generate the RL set: canonical conversation rows with an image path and a `"gold"` answer letter (train split only):

```bash
python3 examples/vlm-2xH200/scripts/prepare_rl_data.py scienceqa \
  --out-dir data/rl_scienceqa --rows 300
```

3. Generate the **multimodal preference set** for Step 2b below: image pairs from `openbmb/RLAIF-V-Dataset` plus grounding video pairs from the canonical video rows:

```bash
python3 examples/vlm-2xH200/scripts/prepare_rl_data.py preference_mm \
  --out-dir data/dpo_mm --n-images 500 \
  --video-jsonl data/prepared/video.train.jsonl --n-video-pairs 500
```

This writes `data/dpo_mm/preference_mm_images.jsonl` (`{"prompt","image","chosen","rejected"}`, one PNG per row under `data/dpo_mm/images/`) and `data/dpo_mm/preference_mm_video.jsonl` (`{"prompt","video","start","end","chosen","rejected"}`). The video pairs are a **grounding construction** (rejected = a different segment's real answer), not a human preference judgement — see the script's own docstring.

**Check it worked.** Both files exist and the RL file has one image path and a gold letter per row:

```text
$ ls -l data/pref/dpo_train.jsonl data/rl_scienceqa/scienceqa_train.jsonl
-rw-r--r-- 1 root root ... data/pref/dpo_train.jsonl
-rw-r--r-- 1 root root ... data/rl_scienceqa/scienceqa_train.jsonl
```

The DPO script reads `data/pref/dpo_train.jsonl`; the GRPO script reads `data/rl_scienceqa/scienceqa_train.jsonl` (train split only) with `gold_key="gold"`.

---

## Step 2 — DPO + LoRA on 2× H200 (text-only, verified)

The two preference/RL trainers ship as ready-made scripts under `examples/vlm-2xH200/scripts/`. They are **real CLIs**: `--help` prints every flag, and each is launched with `torchrun`, with 127.0.0.1 for the rendezvous (IPv6 localhost is not usable on the competition VM image).

1. `examples/vlm-2xH200/scripts/train_dpo.py` — Preference training (DPO, IPO, ORPO, SimPO, CPO) with LoRA over a JSONL of prompt/chosen/rejected triples.

```text
usage: train_dpo.py [-h] --model MODEL --data DATA --output-dir OUTPUT_DIR
                    [--algorithm {dpo,ipo,orpo,simpo,cpo}]
                    [--max-steps MAX_STEPS] [--max-length MAX_LENGTH]
                    [--learning-rate LEARNING_RATE] [--lora-rank LORA_RANK]
                    [--lora-alpha LORA_ALPHA]
```

Only the preference objective is exposed on the CLI (`kto` stays a Python API algorithm); `--output-dir` is the save/scratch directory where `result.json` lands.

2. Launch on both GPUs, from the repository root `FoundationScale/`:

```bash
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 examples/vlm-2xH200/scripts/train_dpo.py --model models/gemma-4-12B-it --data data/pref/dpo_train.jsonl --output-dir runs/dpo --algorithm dpo --max-steps 20 --max-length 4096
```

**Check it worked** (measured, 2× H200, Qwen3.6-27B, bf16, gradient checkpointing, `max_length 4096`, `logprob_micro_batch=1`, 30 steps):

```text
RESULT_JSON {"ok": true, "model": "models/Qwen3.6-27B",
 "max_length": 4096, "logprob_micro_batch": 1, "n_steps_measured": 30,
 "wall_s": 292.5 (9.75 s/step),
 "history_first": {"loss": 0.6931, ...preference_accuracy 0.0, margin 0.0},
 "history_last":  {"loss": 0.6712, ...preference_accuracy 0.75, margin 0.229},
 "ln2": 0.6931471805599453, "step0_minus_ln2": 0.0}
```

- exit code 0, 30/30 steps measured
- loss `0.6931 -> 0.6712`; preference accuracy `0 -> 0.75`; margin `0 -> 0.229`
- `step0_minus_ln2 = 0.0` — the reference-model / LoRA sanity check holds
- peak memory **72.7 GB/GPU** (identical 72.7 GB at `max_length 8192`)
- exact history key names for preference accuracy / margin: `preference_accuracy` and `margin` (next to `loss`), as printed above

Flags: `--model`, `--data`, `--output-dir`, `--algorithm`, `--max-steps`, `--max-length`, `--learning-rate`, `--lora-rank`, `--lora-alpha`. Measured by a fresh user: a 5-step acceptance run is **92 s**; a full run at 30 steps is under 5 minutes.

---

## Step 2b — Multimodal DPO (image + video + text, verified on gemma-4-12B-it and Qwen3.6-27B)

The offline preference trainer now accepts image and/or video rows: declare
`image_column`/`video_column` (`--image-column`/`--video-column` on the CLI)
and a record's declared media is rendered through the SAME processor path
SFT uses (`<image>`/`<video>` markers in `prompt`, media loaded/decoded, the
two sides' completions scored with the shared media attached). A row with
neither column declared trains exactly as Step 2 above — byte-identical.

Row shapes (`prepare_rl_data.py preference_mm`'s own output, Step 1.3):

```text
{"prompt": "<image> ...", "image": "<path>", "chosen": "...", "rejected": "..."}
{"prompt": "<video> ...", "video": "<clip>", "start": 0.0, "end": 2.0, "chosen": "...", "rejected": "..."}
```

1. Image pairs, `gemma-4-12B-it`:

```bash
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29890 \
  examples/vlm-2xH200/scripts/train_dpo.py --model models/gemma-4-12B-it \
  --data data/dpo_mm/preference_mm_images.jsonl --output-dir runs/dpo_mm_image \
  --algorithm dpo --max-steps 20 --max-length 4096 \
  --image-column image
```

2. Video pairs, `gemma-4-12B-it` (16 frames/clip; native video needs more room than a single image, hence `max-length 16384`):

```bash
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29890 \
  examples/vlm-2xH200/scripts/train_dpo.py --model models/gemma-4-12B-it \
  --data data/dpo_mm/preference_mm_video.jsonl --output-dir runs/dpo_mm_video \
  --algorithm dpo --max-steps 20 --max-length 16384 \
  --video-column video --video-frames 16
```

3. Mixed image+video+text, `Qwen3.6-27B`, and an ORPO run on the SAME mixed file (algorithm switching, Step 5). `prepare_rl_data.py preference_mm` writes ONLY `data/dpo_mm/preference_mm_images.jsonl` and `data/dpo_mm/preference_mm_video.jsonl` — there is no ready-made mixed file, so build it first (deterministic shuffle):

```bash
cat data/dpo_mm/preference_mm_images.jsonl data/dpo_mm/preference_mm_video.jsonl | shuf --random-source=<(yes) > data/dpo_mm/preference_mm_mixed.jsonl
```

Add text pairs from `data/pref/dpo_train.jsonl` (Step 1.1) to that file if you want text in the mix.

The Qwen3.6-27B proof uses `--max-length 8192` and 4 video frames per clip (measured longest row: 8,239 tokens):

```bash
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29890 \
  examples/vlm-2xH200/scripts/train_dpo.py --model models/Qwen3.6-27B \
  --data data/dpo_mm/preference_mm_mixed.jsonl --output-dir runs/dpo_mm_mixed \
  --algorithm dpo --max-steps 20 --max-length 8192 \
  --image-column image --video-column video --video-frames 4

torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29890 \
  examples/vlm-2xH200/scripts/train_dpo.py --model models/Qwen3.6-27B \
  --data data/dpo_mm/preference_mm_mixed.jsonl --output-dir runs/orpo_mm_mixed \
  --algorithm orpo --max-steps 10 --max-length 8192 \
  --image-column image --video-column video --video-frames 4
```

**Check it worked** (measured, 2× H200, LoRA r16/α32, FSDP, bf16, gradient checkpointing, `logprob_micro_batch=1`, 20 steps for `gemma-4-12B-it` and `Qwen3.6-27B` DPO, 10 for `Qwen3.6-27B` ORPO; Qwen peak memory is rank 0 only):

| Run | exit | peak GB/GPU | s/step | step-0 loss | last loss | last accuracy | last margin | adapter |
|---|---|---|---|---|---|---|---|---|
| (1) `gemma-4-12B-it`, image pairs | 0 | 28.6 / 27.4 | 10.4 | `0.693147` (`ln 2`) | 1.1044 | 0.25 | -0.582 | loads |
| (2) `gemma-4-12B-it`, video pairs (16 frames) | 0 | 51.2 / 51.4 | 18.8 | `0.693147` (`ln 2`) | 0.0054 | 1.00 | 10.25 | loads |
| (3) `Qwen3.6-27B`, mixed image+video+text, DPO | 0 | 80.2 | 20.6 | `0.693147` (`ln 2`) | 0.5269 | 0.75 | 0.667 | saved |
| (4) `Qwen3.6-27B`, mixed image+video+text, ORPO | 0 | 79.9 | 25.7 | 0.5740 (ORPO adds an NLL term, so not `ln 2`) | 0.9622 | 0.75 | 0.192 | saved |

Three bugs blocked the Qwen3.6-27B rows until they were found and fixed (2026-10-11):

**Bug A (found and fixed): a heterogeneous-batch modality-narrowing crash.** The first mixed-batch run crashed inside the model's own placeholder-count check (`Video features and video tokens do not match, tokens: 0, features: 2400`). ``image_grid_thw``/``video_grid_thw`` count MEDIA ITEMS, not training rows — a batch where only one row carries a video still gets a 1-row ``video_grid_thw``, never a zero-padded one row per training row — so `rl.trainer._narrow_modality_kwargs_by_row`'s own precondition ("one image per prompt," true for the online RL trainer that built it) does not hold for a HETEROGENEOUS image+video+text batch. Fixed by tracking per-row image/video counts (`_EncodedBatch.image_counts`/`video_counts`) and translating a training-row micro-batch range into the matching MEDIA-row range per modality before calling the reused helper — and by OMITTING a modality's keys entirely (never passing an empty tensor) for a micro-batch slice that carries none of it, which a second measured crash (`repeats must have the same size as input along dim`) showed the vision tower does not tolerate either. A THIRD measured crash (`RuntimeError: start (2) + length (1) exceeds dimension size (2)`) showed gemma's own convention (`image_position_ids`/`video_position_ids`, no `*_grid_thw` pair at all) defeated a name-based key classifier; the fix classifies by TENSOR SHAPE against the batch's measured row/image/video counts instead of a hardcoded name list, with a flattened-patch-pair reconciliation pass for Qwen's convention. All three failure modes are pinned as CPU regression tests in `tests/rl/test_preference_trainer_multimodal.py`.

**Bug B (found and fixed): a genuine collective-count deadlock, not a slow step.** `Qwen3.6-27B` DPO/ORPO runs hung past increasingly generous NCCL collective timeouts (600s default, then 30 min, then 90 min configured via `FOUNDATIONSCALE_COLLECTIVE_TIMEOUT_MINUTES` on `init_process_group`). The first py-spy dump of both ranks was misread as a wall-clock imbalance (one rank's Python frame parked in a hot `transformers` RMSNorm call, the SAME frame on repeated samples) because py-spy only samples Python frames and RMSNorm is the hottest call site in any forward, computing or blocked underneath. The NCCL watchdog's OWN flight-recorder dump at the actual timeout told the true story: one rank's innermost frame was FSDP2's `all_gather_single` inside per-layer `unshard`, with `last enqueued` far ahead of `last completed` (prefetched all-gathers piling up unanswered), while the OTHER rank had already left the forward pass entirely and moved on to the next collective. That is a genuine deadlock: `preference_trainer.py`'s `_one_step` sized its logprob micro-batch COUNT (and whether to micro-batch at all) from each rank's own `row_count` with no cross-rank agreement, so an uneven prompt-level shard could give one rank more (or zero) micro-batch forward() calls than its peer for the same step — each forward() re-fires FSDP2's per-layer all-gathers, so a rank short one call leaves its peer's matching all-gather with no partner, forever. `rl/trainer.py` already carries the fix for the identical GRPO-side bug (`agree_max` plus a shared zero-weight pad slice, `_padding_backwards`); `preference_trainer.py` now mirrors it: every rank agrees on the micro-batch slice COUNT (`agree_max`) and on WHETHER to slice at all (`agree_all`), pads up to the busier peer's count with a cheap 1-row dummy forward/backward whose gradient is exactly zero, and only then narrows to its own real rows before concatenating. Two CPU regression tests in `tests/rl/test_preference_trainer_cov.py` pin this: one proves the padded run's optimizer step is bit-for-bit identical to a padding-free control (the padding leaks no gradient), the other proves a rank whose OWN row count would take the single-shot path still slices (and pads) when a peer needs to.

**Bug C (found and fixed): FSDP2 wrapped the Qwen vision blocks separately.** After Bug B's fix, some 20-step runs still stalled for the full 90-minute collective timeout. This was first misread as a very slow row (an earlier version of this page blamed the unoptimized `causal_conv1d` fallback). A `py-spy dump --native` of both ranks during a stall showed the real cause: rank 0 was frozen inside `Qwen3_5VisionBlock.forward` while rank 1, whose row at the same index was text-only, waited at the trainer's own cross-rank barrier. transformers 5.18 lists `Qwen3_5VisionBlock` in Qwen3.6's `_no_split_modules`, so `wrap_fsdp2` gave every vision block its own FSDP2 unit even though tower sharding was off for this media run; an image row on one rank then issued all-gathers the other rank never matched. gemma-4 never hit this because its `_no_split_modules` names only the text decoder layer. `wrap_fsdp2` now folds tower-internal blocks into the root unit when `shard_towers=False` (`tests/rl/test_rl_fsdp_tower_wrap.py`). With it, the 20-step DPO run finished in under 7 minutes at 20.6 s/step (it measured ~90 s/step before), and ORPO ran 10 steps at 25.7 s/step. The rows are short: across all 280 rendered rows the longest is 8,239 tokens, so length was never the cause.

**The control that proves pixels matter** (measured, `gemma-4-12B-it`, single GPU, LoRA r16, 2 DPO steps on the SAME 8 image pairs, same seed so LoRA's random init is identical): step-0 loss is `ln 2 = 0.693147...` in BOTH conditions (policy == reference by construction, independent of pixel content — expected, and itself a check that nothing else differs between the two runs). After one gradient step, real images give loss `0.3253`, solid-black images give loss `0.3145` — DIFFERENT. If they had matched, the vision tower's contribution would be zero and the run would be training on the prompt text alone under a multimodal label. Automated as `validation_campaigns/vlm_2xh200/test_preference_trainer_multimodal_dpo.py`.

Flags added to `train_dpo.py` for this step: `--image-column`, `--video-column`, `--video-frames`, `--video-sampling`, `--video-max-side`.

---

## Step 3 — GRPO + LoRA with images (verified)

The RL rows carry an image and a `"gold"` answer letter. The only built-in verifiable reward is `MCQLetterReward`, configured with `answer_pattern = r"Answer:\s*([A-Z])"`; it parses `"Answer: X"` and **abstains (drops the row)** when no single letter is found.

1. `examples/vlm-2xH200/scripts/train_grpo.py` — GRPO + LoRA RL training (with images) over the RL JSONL rows built above.

```text
usage: train_grpo.py [-h] --model MODEL --data DATA --output-dir OUTPUT_DIR
                     [--max-steps MAX_STEPS] [--max-new-tokens MAX_NEW_TOKENS]
                     [--algorithm {grpo,dr_grpo,dapo,gspo,reinforce_pp,rloo}]
                     [--group-size GROUP_SIZE]
                     [--prompts-per-step PROMPTS_PER_STEP]
                     [--learning-rate LEARNING_RATE] [--lora-rank LORA_RANK]
                     [--lora-alpha LORA_ALPHA]
```

2. Launch on both GPUs (group_size 4, ≥ 4 prompts per step):

```bash
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 examples/vlm-2xH200/scripts/train_grpo.py --model models/gemma-4-12B-it --data data/rl_scienceqa/scienceqa_train.jsonl --output-dir runs/grpo --algorithm grpo --max-steps 20 --max-new-tokens 400 --group-size 4 --prompts-per-step 4
```

To run the 27B instead, swap the model only:

```bash
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 examples/vlm-2xH200/scripts/train_grpo.py --model models/Qwen3.6-27B --data data/rl_scienceqa/scienceqa_train.jsonl --output-dir runs/grpo --algorithm grpo --max-steps 20 --max-new-tokens 400 --group-size 4 --prompts-per-step 4
```

A refused run (for example every GRPO step had identical rewards) **exits 96 with a `REFUSED` message** instead of finishing.

**Check it worked** (measured, 2× H200, 30 steps, `group_size 4`, `MCQLetterReward`, `max_new_tokens 400`; adapters load):

| Model | peak / GPU | steps with a reward | steps abstained |
|---|---|---|---|
| `gemma-4-12B-it` | ~55 GB | 11 / 30 | 19 |
| `Qwen3.6-27B` | ~69 GB | 13 / 30 | 17 |

- exit code 0; the saved LoRA adapter loads
- `kl_contribution` at the first step must be exactly `0` — the reference-model sanity check (same model, adapter off, no second copy in VRAM)
- abstained steps are **expected**: on hard questions all 4 samples kept reasoning and ran out of `max_new_tokens` before writing `"Answer: X"`; `MCQLetterReward` then drops the row and the step produces no reward
- wall time and per-step reward stats land in `runs/grpo/result.json` (`reward_mean`, `reward_std`, `reward_min`, `reward_max`, `kl_contribution` per step); measured by a fresh user: a 5-step acceptance run is **198 s**

Flags: `--model`, `--data`, `--output-dir`, `--algorithm`, `--max-steps`, `--max-new-tokens`, `--group-size`, `--prompts-per-step`, `--learning-rate`, `--lora-rank`, `--lora-alpha`.

### GRPO with images on the Qwen models (status: working, after three fixes)

GRPO with images on `Qwen3.6-27B` and `Qwen3.6-35B-A3B` used to be broken by three vision-specific bugs. All three are **fixed on the current branch**:

1. the RL trainer loaded the **text-only class** for the Qwen models (the vision path never reached the RL trainer);
2. **prompt-length `mm_token_type_ids`** were forwarded together with the prompt+completion ids;
3. **Qwen's flattened patch tensor was sliced by row** in group expansion and micro-batching.

After the fixes **both models run GRPO with images 10 steps without error**: peak **60.3 GB/GPU** (`Qwen3.6-27B`) and **82.2 GB/GPU** (`Qwen3.6-35B-A3B`), on the 2× H200 run recipe used throughout this chapter (LoRA r16/α32, `sharding="ddp"` — hard-coded in `train_grpo.py` — bf16, activation checkpointing; `train_grpo.py` has no `--fused-loss` option). The gemma models never had these problems: `gemma-4-26B-A4B-it` and `gemma-4-31B-it` each run 20 steps of GRPO with images, exit 0. See the algorithm × model matrix in Step 5 and the Troubleshooting entry if a Qwen run fails.

---

## Step 4 — Get the reward you expect: budget the completions

Measured per-completion parse rate for `"Answer: X"` with reasoning models:

| `max_new_tokens` | completions parsed |
|---|---|
| 64 | 0 % |
| 200 | 25 % |
| 400 | 85 % |

Advice baked into the example command above: **set `max_new_tokens` to 400 or more for reasoning models**, and put an explicit *"end with `Answer: X`"* instruction in the prompt. Otherwise most groups abstain and you train on a fraction of the steps.

---

## Step 5 — Switching algorithms

One field, one dataset format:

```python
# offline preference training (PreferenceTrainer) — text, image and/or video rows
cfg = PreferenceTrainConfig(..., algorithm="ipo")  # dpo | ipo | kto | orpo | simpo | cpo
# add image_column="image" / video_column="video" (+ video_frames) to train on declared media
```

```python
# online RL (RLTrainer) — image rows OK, no video
cfg = RLTrainConfig(..., algorithm="rloo")  # grpo | dr_grpo | dapo | gspo |
# reinforce_baseline | reinforce_pp | rloo | raft |
# best_of_n | online_dpo | iterative_dpo | ppo
```

The 12 RL algorithms share the same `algorithm=` field (CLI: `--algorithm`); per-algorithm hyperparameter names beyond the fields shown in `examples/vlm-2xH200/scripts/train_grpo.py` are not exposed — `--learning-rate`, `--lora-rank` and `--lora-alpha` are the only extra tuning flags, and every RL objective consumes the same fields.

### Sharding and LoRA memory (2× H200)

- The GRPO/RL recipe is **DDP + LoRA + activation checkpointing, exactly as in `train_grpo.py`**: `sharding="ddp"` is **hard-coded** in that script, because repeated generation under FSDP is not yet reliable. There is no `--fused-loss` option on `train_grpo.py`.
- The shipped DPO preference recipe runs with `sharding="fsdp"` (its measured 72.7 GB/GPU is under that setting).
- `sharding="ddp"` is the validated choice for preference training with LoRA as well — it fits, with the peaks measured above. Use `ddp` when in doubt.
- Keep `logprob_micro_batch=1` for long preference rows: without it the peak was **125 GB** at `max_length 4096` (fp32 vocab logits of a long row); with it, 72.7 GB at 4096 and the same 72.7 GB at 8192.

### Rollout generation is now safe with gradient checkpointing

Generation runs in eval mode with the KV cache even when `gradient_checkpointing=True`. An earlier bug corrupted rollouts after the first token; it is fixed and verified (identical 32-token greedy rollout with checkpointing on/off). Training and generation can therefore both use gradient checkpointing.

### Algorithm × model results (measured sweep, 2 × H200)

Every cell below is a measured run of the **20-step acceptance sweep**: 2× H200, LoRA `adapter_rank=16` / `adapter_alpha=32`, bf16, activation checkpointing. Preference/DPO rows ran under the DPO recipe (`sharding="fsdp"`, `--fused-loss liger` on `train_dpo.py`); the GRPO/RL rows ran under the GRPO recipe (`sharding="ddp"` hard-coded in `train_grpo.py`, DDP + LoRA + activation checkpointing — no `--fused-loss` option there). Unless a cell says otherwise, each run trained **20 steps and exited 0**. A dash (—) = not part of this sweep.

| `algorithm=` | `gemma-4-12B-it` | `gemma-4-26B-A4B-it` | `gemma-4-31B-it` | `Qwen3.6-27B` | `Qwen3.6-35B-A3B` |
|---|---|---|---|---|---|
| `dpo` + LoRA (text only) | 20 steps, exit 0 | 20 steps, exit 0 | 20 steps, exit 0 | 20 steps, exit 0 | 20 steps, exit 0 |
| `ipo` (text only) | — | — | — | 20 steps, exit 0 | — |
| `simpo` (text only) | — | — | — | 20 steps, exit 0 | — |
| `cpo` (text only) | — | — | — | 20 steps, exit 0 | — |
| `orpo` (text only) | — | — | — | 20 steps, exit 0 | — |
| `grpo`, with images | 20 steps, exit 0 | 20 steps, exit 0 | 20 steps, exit 0 | 10 steps with images, no error, peak 60.3 GB/GPU † | 10 steps with images, no error, peak 82.2 GB/GPU † |
| `dr_grpo`, with images | 20 steps, exit 0 | — | — | — | — |
| `dapo`, with images | 20 steps, exit 0 | — | — | — | — |
| `gspo`, with images | 20 steps, exit 0 | — | — | — | — |
| `reinforce_pp` (REINFORCE++), with images | 20 steps, exit 0 | — | — | — | — |
| `rloo`, with images | 20 steps, exit 0 | — | — | — | — |

† GRPO with images on the Qwen models runs only **after** the three vision fixes (see Step 3): text-only class loaded by the RL trainer; prompt-length `mm_token_type_ids` forwarded with prompt+completion ids; Qwen's flattened patch tensor sliced by row in group expansion and micro-batching. Before the fixes the run failed; after them both run 10 steps with images without error.

Not in this sweep: `kto`, `raft`, `best_of_n`, `online_dpo`, `iterative_dpo`, `ppo` (registered, but no measured numbers to report here). Modality rules: preference rows take text, image and/or video (declare `image_column`/`video_column`, Step 2b); `RLTrainer` still refuses video rows (images only).

---

## The GRPO signal: zero-variance groups and skipped steps (measured)

GRPO turns rewards into advantages **inside a group**. If every sample of a prompt gets the same reward, the advantage is exactly `0`, the step carries no gradient and is **skipped**. On ScienceQA this happens often — the measured causes are all-identical rewards in the group:

- the model gets **all 4 samples right**, or
- **none of the 4 samples finishes** within `max_new_tokens`

— so one group's rewards are all identical and there is no signal to train on.

If **every** step of a run is skipped, the run is refused outright: **exit 96, `REFUSED`** ("vacuous run" = no one step ever produced a trainable signal).

**Measured** (2× H200, `gemma-4-12B-it`, GRPO with images, `group_size 4`, 4 prompts per step, 20 steps attempted): only **3 of 20 steps** produced a trainable signal — the remaining steps were skipped on identical group rewards. Exit codes 0 for the run itself; 3/20 is the kind of yield you should expect from `group_size 4` / 4 prompts per step on ScienceQA.

### Remedies

- **larger `--group-size` (8)** (`group_size=8`) — more samples per prompt, so a group is more likely to contain both right and wrong completions instead of 4 identical rewards;
- **more `--prompts-per-step`** (`prompts_per_step`) — more independent groups per step means more chance that at least one of them has variance;
- **harder questions matched to the model** — prompts the model already solves at 4/4 carry no signal; move to questions it has not mastered;
- **`--max-new-tokens 400+` for reasoning models** (the Step 4 advice) — when samples have room to finish, groups stop all-tieing on "none finished".

Start with a larger group and more prompts per step (the cheapest change), then tune the question difficulty and the completion budget.

---

## Difficulty filtering for GRPO, and what it did (measured)

GRPO learns only from groups whose rewards differ. Before training, measure each question's
pass rate with the base model and keep the ones it gets right *sometimes*:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 \
  vllm-env/bin/python examples/vlm-2xH200/scripts/filter_by_pass_rate.py \
  --model models/gemma-4-12B-it --data data/rl_scienceqa/scienceqa_train.jsonl \
  --out-keep data/rl_scienceqa/keep.jsonl --out-stats data/rl_scienceqa/filter_stats.json \
  --samples 8 --temperature 1.0 --max-tokens 400 --min-rate 0.125 --max-rate 0.875 --tp 1
```

`vllm-env` here is the separate venv for vLLM — it is created in [chapter 05 §4a](05_evaluation.md); create it first if you haven't.

Then split `keep.jsonl` into train and held-out files with `prepare_data.py split`, and train GRPO
on the train part only.

Measured on gemma-4-12B-it with 3,000 ScienceQA train questions: 72.8% were answered correctly
in all 8 samples, 4.4% never, so 686 were kept (480 train / 206 held out). With the filtered set,
every GRPO step produced an update (50 of 50, versus 12 of 30 unfiltered). But neither recipe
improved held-out accuracy:

| GRPO recipe (group 8, 8 prompts/step, LoRA r16) | KL at end | held-out greedy accuracy | unparsable |
|---|---|---|---|
| base model | - | 119 / 206 (57.8%) | 9.7% |
| lr 1e-4, 50 steps | ~0.0003 | 117 / 206 (56.8%) | 17.0% |
| lr 5e-4, 150 steps | ~0.02 | 44 / 206 (21.4%) | 41.7% |

At 1e-4 the policy barely moved. At 5e-4 it drifted away from ending with "Answer: X" inside the
token budget (format collapse), while the training reward stayed flat and looked fine. Lessons:
filter for difficulty first; always track held-out accuracy, the unparsable rate and KL, never
the training reward alone; and sweep the learning rate between these two points before a long run.

## Troubleshooting

**Exit code 96 on the preference trainer.** Your JSONL has a row carrying an image or a video field that is NOT declared: pass `--image-column`/`--video-column` naming that field (Step 2b), or strip the field if the row really is text-only. The trainer never guesses a media column.

**The RL trainer refuses my video rows.** Online RL supports images but not video. Use SFT for video.

**CUDA OOM in preference training (you're seeing ~125 GB/GPU at `max_length 4096`).** Set `logprob_micro_batch=1`; long rows otherwise materialize fp32 vocab logits. Measured peak with it: 72.7 GB/GPU at 4096 and 8192.

**`n_steps_abstained` equals `n_steps_attempted` — no reward, ever.** Your completions never reach `"Answer: X"`. Measured parse rate: 0 % at `max_new_tokens 64`, 25 % at 200, 85 % at 400. Raise `max_new_tokens` to 400+, add "end with `Answer: X`" to the prompt. Remember `MCQLetterReward` drops the row whenever it finds no single letter.

**DPO step-0 loss is not exactly `ln 2 = 0.6931`, or GRPO step-0 KL is not exactly `0`.** The reference model must be the same model with the adapter switched off. Check that `adapter="lora"` and the adapter fields are set and that nothing loads a second (or a diverged copy of the) model. The DPO run writes `result.json` into `--output-dir` precisely for this check (measured `step0_minus_ln2 = 0.0`).

**Rollouts look corrupted after the first token with gradient checkpointing on.** You are on a build older than the generate-in-eval-mode fix. Update FoundationScale and re-run: greedy 32-token rollouts must be identical with checkpointing on and off.

**Generation hangs or behaves oddly under FSDP on the RL trainer.** You are not on the shipped `train_grpo.py` path — it hard-codes `sharding="ddp"` precisely because repeated generation under FSDP is not yet reliable. Run DDP (the validated `train_grpo.py` default) with LoRA on 2× H200.

**Only some steps of 30 produce a reward (11/30 or 13/30 is the measured norm).** Expected on hard ScienceQA questions with `group_size 4`: all 4 samples keep reasoning and hit `max_new_tokens`. This is abstention, not a bug.

**GRPO with images on a Qwen model fails (`mm_token_type_ids` mismatch, patch-tensor slicing, or a run that silently trains text only).** Three vision bugs, all fixed on the current branch: the text-only class was loaded by the RL trainer; prompt-length `mm_token_type_ids` were forwarded with the prompt+completion ids; Qwen's flattened patch tensor was sliced by row in group expansion and micro-batching. After the fixes `Qwen3.6-27B` and `Qwen3.6-35B-A3B` run 10 steps of GRPO with images without error (peak 60.3 / 82.2 GB per GPU, 2× H200, LoRA r16/α32).

**Exit 96 with a `REFUSED` ("vacuous run") message, or almost every step is skipped.** Every group scored identical rewards (all 4 samples right, or none finished within `max_new_tokens`), so every advantage was zero and every step was skipped; the trainer refuses a run with no trainable step at all. Measured here: `gemma-4-12B-it` produced 3 of 20 trainable steps with `group_size 4` / 4 prompts per step. Remedies: `--group-size 8`, more `--prompts-per-step`, harder questions matched to the model, `--max-new-tokens 400+` for reasoning models (see "The GRPO signal" section above).

**Edit the `algorithm` field name to `objective` by habit?** The shipped scripts take `algorithm="dpo"` / `algorithm="grpo"` on `PreferenceTrainConfig` / `RLTrainConfig` (`--algorithm dpo` / `--algorithm grpo` on the CLI); there is no separate `objective` field to set — `objective` is only a legacy spelling found in some older configs.

Next: chapter 05 — [Evaluation](05_evaluation.md).

**Next:** [05 — Evaluation](05_evaluation.md)