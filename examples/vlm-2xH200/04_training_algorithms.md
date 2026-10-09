# 04 — Training algorithms: SFT, DPO-family preference training and GRPO-family RL

FoundationScale registers 18 algorithms behind three front doors. Everything you learned in chapters 02–03 still applies (data loading, LoRA, bf16, acceptance runs); this chapter shows how to run preference training and RL on your 2× H200 node and what the measured numbers look like.

| Front door | Trainer / CLI | `algorithm=` values | Modality support |
|---|---|---|---|
| SFT | `foundationscale-train` CLI (chapter 03) | `sft` | image + video + text |
| Offline preference | `PreferenceTrainer` | `dpo`, `ipo`, `kto`, `orpo`, `simpo`, `cpo` | **TEXT ONLY** — rows containing an image or a video are refused (exit 96) |
| Online RL | `RLTrainer` | `grpo`, `dr_grpo`, `dapo`, `gspo`, `reinforce_baseline`, `reinforce_pp`, `rloo`, `raft`, `best_of_n`, `online_dpo`, `iterative_dpo`, `ppo` | **images yes, video refused** |

Switching algorithm is a one-line config change: the `algorithm` field of `PreferenceTrainConfig` / `RLTrainConfig` (the `objective` field name appears in some configs; TODO(verify) in which). Everything else — data format, sharding, LoRA fields — stays the same.

Two runbook properties you get for free on every algorithm:

- **LoRA field names mirror SFT**: `adapter="lora"`, `adapter_rank`, `adapter_alpha`, `adapter_dropout`, `adapter_targets`. The reference model is *the same model with the adapter switched off*, so no second copy is ever loaded.
- **Sanity checks that must hold exactly**: at step 0 the DPO loss is exactly `ln 2 = 0.6931` and the GRPO KL is exactly `0`. If they don't, stop and see Troubleshooting.

---

## Step 1 — Prepare the two datasets

`prepare_rl_data.py` emits both formats. Run from the repo root (`/workspace/fs-vlm`, the parent of `models/`; TODO(verify) if your checkout lives elsewhere).

1. Generate the text-only preference set (`{"prompt","chosen","rejected"}` JSONL rows):

```bash
cd /workspace/fs-vlm
python3 examples/vlm-2xH200/scripts/prepare_rl_data.py preference \
  --out data/pref/dpo_train.jsonl --rows 2000
```

2. Generate the RL set: canonical conversation rows with an image path and a `"gold"` answer letter (train split only):

```bash
python3 examples/vlm-2xH200/scripts/prepare_rl_data.py scienceqa \
  --out-dir data/rl_scienceqa --rows 300
```

**Check it worked.** Both files exist and the RL file has one image path and a gold letter per row:

```text
$ ls -l data/pref/dpo_train.jsonl data/rl_scienceqa/scienceqa_train.jsonl
-rw-r--r-- 1 root root ... data/pref/dpo_train.jsonl
-rw-r--r-- 1 root root ... data/rl_scienceqa/scienceqa_train.jsonl
```

The DPO script reads `data/pref/dpo_train.jsonl`; the GRPO script reads `data/rl_scienceqa/scienceqa_train.jsonl` (train split only) with `gold_key="gold"`.

---

## Step 2 — DPO + LoRA on 2× H200 (text-only, verified)

The two preference/RL trainers are a **Python API**, not a CLI: ship a small script and launch it with `torchrun`.

1. Save this as `scripts/train_dpo.py` (verified as shipped):

```python
#!/usr/bin/env python3
"""GPU proof (a): DPO + LoRA, text-only, 2x H200, FSDP2, bf16, gradient
checkpointing. Run under torchrun --nproc_per_node 2.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

from foundationscale.rl.preference_trainer import PreferenceTrainConfig, PreferenceTrainer

MODEL = os.environ.get("DPO_MODEL", "/workspace/fs-vlm/models/Qwen3.6-27B")
DATASET = "/workspace/fs-vlm/data/pref/dpo_train.jsonl"
SAVE_DIR = os.environ.get("DPO_SAVE_DIR", "/workspace/fs-vlm/runs/dpo_lora_proof")
MAX_STEPS = int(os.environ.get("DPO_MAX_STEPS", "30"))
MAX_LENGTH = int(os.environ.get("DPO_MAX_LENGTH", "4096"))
LOGPROB_MICRO_BATCH = int(os.environ.get("DPO_MICRO_BATCH", "1"))


def _is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def main() -> int:
    cfg = PreferenceTrainConfig(
        model=MODEL,
        dataset=DATASET,
        algorithm="dpo",
        learning_rate=1e-4,
        pairs_per_step=4,
        max_steps=MAX_STEPS,
        max_length=MAX_LENGTH,
        logprob_micro_batch=LOGPROB_MICRO_BATCH,
        sharding="fsdp",
        gradient_checkpointing=True,
        adapter="lora",
        adapter_rank=16,
        adapter_alpha=32.0,
        save_dir=SAVE_DIR,
        save_every=0,
        seed=0,
    )
    if _is_rank0():
        print(f"[run_dpo_lora_proof] model={MODEL} max_steps={MAX_STEPS}", file=sys.stderr)
    trainer = PreferenceTrainer(cfg)
    t0 = time.time()
    reports = trainer.run()
    t1 = time.time()

    if _is_rank0():
        os.makedirs(SAVE_DIR, exist_ok=True)
        history = trainer.history
        result = {
            "ok": True,
            "model": MODEL,
            "max_length": MAX_LENGTH,
            "logprob_micro_batch": LOGPROB_MICRO_BATCH,
            "n_steps_measured": len(reports),
            "wall_s": t1 - t0,
            "s_per_step": (t1 - t0) / max(1, len(reports)),
            "history_first": history[0] if history else None,
            "history_last": history[-1] if history else None,
            "ln2": math.log(2),
            "step0_minus_ln2": (history[0]["loss"] - math.log(2)) if history else None,
        }
        out_path = os.path.join(SAVE_DIR, "result.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        print("RESULT_JSON " + json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

2. Launch on both GPUs:

```bash
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 \
  scripts/train_dpo.py
```

**Check it worked** (measured, 2× H200, Qwen3.6-27B, bf16, gradient checkpointing, `max_length 4096`, `logprob_micro_batch=1`, 30 steps):

```text
RESULT_JSON {"ok": true, "model": "/workspace/fs-vlm/models/Qwen3.6-27B",
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
- exact history key names for preference accuracy / margin: TODO(verify)

Environment overrides: `DPO_MODEL`, `DPO_SAVE_DIR`, `DPO_MAX_STEPS`, `DPO_MAX_LENGTH`, `DPO_MICRO_BATCH`. Full run at 30 steps is under 5 minutes.

---

## Step 3 — GRPO + LoRA with images (verified)

The RL rows carry an image and a `"gold"` answer letter. The only built-in verifiable reward is `MCQLetterReward`, configured here with `answer_pattern = r"Answer:\s*([A-Z])"`; it parses `"Answer: X"` and **abstains (drops the row)** when no single letter is found.

1. Save this as `scripts/train_grpo.py` (verified as shipped):

```python
#!/usr/bin/env python3
"""GPU re-proof (item 4): GRPO + LoRA + images, after the generate()-eval-mode
fix (item 1) and the FSDP2-vision-tower fix (item 2). sharding='fsdp',
gradient_checkpointing=True, group_size=4, >=4 prompts/step, 30 steps.
Uses MCQLetterReward.answer_pattern = "Answer:\\s*([A-Z])" against the
"Answer: X" instruction format. Run under torchrun --nproc_per_node 2.
"""

from __future__ import annotations

import json
import os
import sys
import time

from foundationscale.rl.trainer import RLTrainConfig, RLTrainer

MODEL = os.environ.get("GRPO_MODEL", "/workspace/fs-vlm/models/gemma-4-12B-it")
DATASET = "/workspace/fs-vlm/data/rl_scienceqa/scienceqa_train.jsonl"
SAVE_DIR = os.environ.get("GRPO_SAVE_DIR", "/workspace/fs-vlm/runs/grpo_lora_proof_v2")
MAX_STEPS = int(os.environ.get("GRPO_MAX_STEPS", "30"))
PROMPTS_PER_STEP = int(os.environ.get("GRPO_PROMPTS_PER_STEP", "8"))
ANSWER_PATTERN = r"Answer:\s*([A-Z])"


def _is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def main() -> int:
    cfg = RLTrainConfig(
        model=MODEL,
        dataset=DATASET,
        gold_key="gold",
        algorithm="grpo",
        group_size=4,
        answer_pattern=ANSWER_PATTERN,
        learning_rate=1e-4,
        max_steps=MAX_STEPS,
        max_new_tokens=400,
        temperature=1.0,
        top_p=0.95,
        prompts_per_step=PROMPTS_PER_STEP,
        logprob_micro_batch=2,
        sharding="ddp",
        gradient_checkpointing=True,
        adapter="lora",
        adapter_rank=16,
        save_dir=SAVE_DIR,
        save_every=0,
        seed=0,
    )
    if _is_rank0():
        print(
            f"[run_grpo_lora_v2] model={MODEL} max_steps={MAX_STEPS} "
            f"prompts_per_step={PROMPTS_PER_STEP} sharding=ddp "
            f"gradient_checkpointing=True answer_pattern={ANSWER_PATTERN!r}",
            file=sys.stderr,
        )
    trainer = RLTrainer(cfg)
    t0 = time.time()
    reports = trainer.run()
    t1 = time.time()

    if _is_rank0():
        os.makedirs(SAVE_DIR, exist_ok=True)
        per_step = []
        for r in reports:
            entry = {"step": r.step, "loss": r.loss.loss, "rows": r.rows}
            if r.reward_stats is not None:
                entry["reward_mean"] = r.reward_stats.mean
                entry["reward_std"] = r.reward_stats.std
                entry["reward_min"] = r.reward_stats.minimum
                entry["reward_max"] = r.reward_stats.maximum
            kl_component = next(
                (c.contribution for c in r.loss.components if "kl" in c.name.lower()), None
            )
            if kl_component is not None:
                entry["kl_contribution"] = kl_component
            per_step.append(entry)
        result = {
            "ok": True,
            "model": MODEL,
            "prompts_per_step": PROMPTS_PER_STEP,
            "n_steps_attempted": MAX_STEPS,
            "n_steps_measured": len(reports),
            "n_steps_abstained": MAX_STEPS - len(reports),
            "wall_s": t1 - t0,
            "s_per_step": (t1 - t0) / max(1, len(reports)),
            "per_step": per_step,
            "first5": per_step[:5],
            "last5": per_step[-5:],
        }
        out_path = os.path.join(SAVE_DIR, "result.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        print("RESULT_JSON " + json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

2. Launch on both GPUs (group_size 4, ≥ 4 prompts per step):

```bash
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 \
  scripts/train_grpo.py
```

To run the 27B instead, swap the model only:

```bash
GRPO_MODEL=/workspace/fs-vlm/models/Qwen3.6-27B \
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 \
  scripts/train_grpo.py
```

**Check it worked** (measured, 2× H200, 30 steps, `group_size 4`, `MCQLetterReward`, `max_new_tokens 400`; adapters load):

| Model | peak / GPU | steps with a reward | steps abstained |
|---|---|---|---|
| `gemma-4-12B-it` | ~55 GB | 11 / 30 | 19 |
| `Qwen3.6-27B` | ~69 GB | 13 / 30 | 17 |

- exit code 0; the saved LoRA adapter loads
- `kl_contribution` at the first step must be exactly `0` — the reference-model sanity check (same model, adapter off, no second copy in VRAM)
- abstained steps are **expected**: on hard questions all 4 samples kept reasoning and ran out of `max_new_tokens` before writing `"Answer: X"`; `MCQLetterReward` then drops the row and the step produces no reward
- wall time and per-step fields with measured reward stats: reproduced in `runs/grpo_lora_proof_v2/result.json` (TODO(verify) exact printed JSON)

Environment overrides: `GRPO_MODEL`, `GRPO_SAVE_DIR`, `GRPO_MAX_STEPS`, `GRPO_PROMPTS_PER_STEP`.

### GRPO with images on the Qwen models (status: working, after three fixes)

GRPO with images on `Qwen3.6-27B` and `Qwen3.6-35B-A3B` used to be broken by three vision-specific bugs. All three are **fixed on the current branch**:

1. the RL trainer loaded the **text-only class** for the Qwen models (the vision path never reached the RL trainer);
2. **prompt-length `mm_token_type_ids`** were forwarded together with the prompt+completion ids;
3. **Qwen's flattened patch tensor was sliced by row** in group expansion and micro-batching.

After the fixes **both models run GRPO with images 10 steps without error**: peak **60.3 GB/GPU** (`Qwen3.6-27B`) and **82.2 GB/GPU** (`Qwen3.6-35B-A3B`), on the 2× H200 run recipe used throughout this chapter (LoRA r16/α32, FSDP, bf16, activation checkpointing, `--fused-loss liger`). The gemma models never had these problems: `gemma-4-26B-A4B-it` and `gemma-4-31B-it` each run 20 steps of GRPO with images, exit 0. See the algorithm × model matrix in Step 5 and the Troubleshooting entry if a Qwen run fails.

---

## Step 4 — Get the reward you expect: budget the completions

Measured per-completion parse rate for `"Answer: X"` with reasoning models:

| `max_new_tokens` | completions parsed |
|---|---|
| 64 | 0 % |
| 200 | 25 % |
| 400 | 85 % |

Advice baked into the verified script above: **set `max_new_tokens` to 400 or more for reasoning models**, and put an explicit *"end with `Answer: X`"* instruction in the prompt. Otherwise most groups abstain and you train on a fraction of the steps.

---

## Step 5 — Switching algorithms

One field, one dataset format:

```python
# offline preference training (PreferenceTrainer) — text-only rows
cfg = PreferenceTrainConfig(..., algorithm="ipo")  # dpo | ipo | kto | orpo | simpo | cpo
```

```python
# online RL (RLTrainer) — image rows OK, no video
cfg = RLTrainConfig(..., algorithm="rloo")  # grpo | dr_grpo | dapo | gspo |
# reinforce_baseline | reinforce_pp | rloo | raft |
# best_of_n | online_dpo | iterative_dpo | ppo
```

The 12 RL algorithms share the same `algorithm=` field; per-algorithm hyperparameter names beyond the fields shown in `scripts/train_grpo.py`: TODO(verify).

### Sharding and LoRA memory (2× H200)

- `sharding="ddp"` is the validated choice for RL and preference training with LoRA — it fits, with the peaks measured above.
- `sharding="fsdp"` works for RL **training**, but repeated generation under FSDP is not yet reliable. The verified DPO script ships with `sharding="fsdp"` (its measured 72.7 GB/GPU is under that setting); use `ddp` when in doubt.
- Keep `logprob_micro_batch=1` for long preference rows: without it the peak was **125 GB** at `max_length 4096` (fp32 vocab logits of a long row); with it, 72.7 GB at 4096 and the same 72.7 GB at 8192.

### Rollout generation is now safe with gradient checkpointing

Generation runs in eval mode with the KV cache even when `gradient_checkpointing=True`. An earlier bug corrupted rollouts after the first token; it is fixed and verified (identical 32-token greedy rollout with checkpointing on/off). Training and generation can therefore both use gradient checkpointing.

### Algorithm × model results (measured sweep, 2 × H200)

Every cell below is a measured run of the **20-step acceptance sweep**: 2× H200, LoRA `adapter_rank=16` / `adapter_alpha=32`, FSDP, bf16, activation checkpointing, `--fused-loss liger`. Unless a cell says otherwise, each run trained **20 steps and exited 0**. A dash (—) = not part of this sweep.

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

Not in this sweep: `kto`, `raft`, `best_of_n`, `online_dpo`, `iterative_dpo`, `ppo` (registered, but no measured numbers to report here). Modality rules do not change with the algorithm: preference rows stay text-only and `RLTrainer` still refuses video rows.

---

## The GRPO signal: zero-variance groups and skipped steps (measured)

GRPO turns rewards into advantages **inside a group**. If every sample of a prompt gets the same reward, the advantage is exactly `0`, the step carries no gradient and is **skipped**. On ScienceQA this happens often — the measured causes are all-identical rewards in the group:

- the model gets **all 4 samples right**, or
- **none of the 4 samples finishes** within `max_new_tokens`

— so one group's rewards are all identical and there is no signal to train on.

If **every** step of a run is skipped, the run is refused outright: **exit 96, "vacuous run"** (= no one step ever produced a trainable signal).

**Measured** (2× H200, `gemma-4-12B-it`, GRPO with images, `group_size 4`, 4 prompts per step, 20 steps attempted): only **3 of 20 steps** produced a trainable signal — the remaining steps were skipped on identical group rewards. Exit codes 0 for the run itself; 3/20 is the kind of yield you should expect from `group_size 4` / 4 prompts per step on ScienceQA.

### Remedies

- **larger `--group-size` (8)** (`group_size=8`) — more samples per prompt, so a group is more likely to contain both right and wrong completions instead of 4 identical rewards;
- **more `--prompts-per-step`** (`prompts_per_step`) — more independent groups per step means more chance that at least one of them has variance;
- **harder questions matched to the model** — prompts the model already solves at 4/4 carry no signal; move to questions it has not mastered;
- **`--max-new-tokens 400+` for reasoning models** (the Step 4 advice) — when samples have room to finish, groups stop all-tieing on "none finished".

Start with a larger group and more prompts per step (the cheapest change), then tune the question difficulty and the completion budget.

---

## Troubleshooting

**Exit code 96 on the preference trainer.** Your JSONL has a row with an image or a video; `PreferenceTrainer` is text only (`{"prompt","chosen","rejected"}`). Strip multimodal rows or use SFT (chapters 02–03).

**The RL trainer refuses my video rows.** Online RL supports images but not video. Use SFT for video.

**CUDA OOM in preference training (you're seeing ~125 GB/GPU at `max_length 4096`).** Set `logprob_micro_batch=1`; long rows otherwise materialize fp32 vocab logits. Measured peak with it: 72.7 GB/GPU at 4096 and 8192.

**`n_steps_abstained` equals `n_steps_attempted` — no reward, ever.** Your completions never reach `"Answer: X"`. Measured parse rate: 0 % at `max_new_tokens 64`, 25 % at 200, 85 % at 400. Raise `max_new_tokens` to 400+, add "end with `Answer: X`" to the prompt. Remember `MCQLetterReward` drops the row whenever it finds no single letter.

**DPO step-0 loss is not exactly `ln 2 = 0.6931`, or GRPO step-0 KL is not exactly `0`.** The reference model must be the same model with the adapter switched off. Check that `adapter="lora"` and the adapter fields are set and that nothing loads a second (or a diverged copy of the) model. The verified DPO script prints `step0_minus_ln2` into `result.json` precisely for this check.

**Rollouts look corrupted after the first token with gradient checkpointing on.** You are on a build older than the generate-in-eval-mode fix. Update FoundationScale and re-run: greedy 32-token rollouts must be identical with checkpointing on and off.

**Generation hangs or behaves oddly under FSDP on the RL trainer.** Repeated generation under FSDP is not yet reliable; switch to `sharding="ddp"` (validated with LoRA on 2× H200).

**Only some steps of 30 produce a reward (11/30 or 13/30 is the measured norm).** Expected on hard ScienceQA questions with `group_size 4`: all 4 samples keep reasoning and hit `max_new_tokens`. This is abstention, not a bug.

**GRPO with images on a Qwen model fails (`mm_token_type_ids` mismatch, patch-tensor slicing, or a run that silently trains text only).** Three vision bugs, all fixed on the current branch: the text-only class was loaded by the RL trainer; prompt-length `mm_token_type_ids` were forwarded with the prompt+completion ids; Qwen's flattened patch tensor was sliced by row in group expansion and micro-batching. After the fixes `Qwen3.6-27B` and `Qwen3.6-35B-A3B` run 10 steps of GRPO with images without error (peak 60.3 / 82.2 GB per GPU, 2× H200, LoRA r16/α32).

**Exit 96 with "vacuous run", or almost every step is skipped.** Every group scored identical rewards (all 4 samples right, or none finished within `max_new_tokens`), so every advantage was zero and every step was skipped; the trainer refuses a run with no trainable step at all. Measured here: `gemma-4-12B-it` produced 3 of 20 trainable steps with `group_size 4` / 4 prompts per step. Remedies: `--group-size 8`, more `--prompts-per-step`, harder questions matched to the model, `--max-new-tokens 400+` for reasoning models (see "The GRPO signal" section above).

**Edit `sharding` field name to `objective` by habit?** The verified scripts take `algorithm="dpo"` / `algorithm="grpo"` on `PreferenceTrainConfig` / `RLTrainConfig`; other config field names for the objective: TODO(verify).

Next: chapter 05 — TODO(verify) title and link.

**Next:** [05 — Evaluation](05_evaluation.md)