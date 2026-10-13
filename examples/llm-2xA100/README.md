# Text-only LLM LoRA SFT on 2x A100 with FoundationScale

One end-to-end recipe: take a chat LLM off Hub, take a public chat SFT dataset, adapt a shared
multimodal pipeline to text-only data, train a LoRA adapter on 2 GPUs, and evaluate it honestly —
held-out negative log-likelihood of the assistant turns, base model vs. adapter, nothing else.
The model is **`Qwen/Qwen2.5-7B-Instruct`** (`model_type` `qwen2`), the data is
**`HuggingFaceH4/ultrachat_200k`** (first 20k train conversations, first 1k test conversations). Sibling
tutorials: [VLM](../vlm-2xH200/) (this one reuses its data validator and eval script), [speech](../speech-2xH200/),
[A100 environment](../A100_2GPU.md).

Every number in the tables is **MEASURED (2026-10-12, 2x A100-SXM4-80GB)** on a shared 8-GPU node with
CUDA pinned to two GPUs. Things not actually run are marked **UNMEASURED**. Read the loss numbers
for what they are: in-distribution (UltraChat test vs. UltraChat train). They prove the pipeline
learns, not that the model is good in general.

## What you get: measured results

Recipe: LoRA r16/alpha32 on `q,k,v,o,gate,up,down`; 4096-token context (overlong rows dropped, never
truncated); FSDP over 2 GPUs; bf16; sdpa; gradient checkpointing; per-device batch 1 × grad-accum 8 ×
2 GPUs = global batch 16; lr 1e-4 cosine, warmup 20; 500 steps; no fused loss.

| step | wall | s/step | train tokens/s | peak GB/GPU | train_loss | exit | verdict |
|---|---|---|---|---|---|---|---|
| 500 steps (save at 250, 500) | 2341 s (`train_runtime`) | 4.68 | ~4,100 | 21.1 | 0.9208 | 0 | PASS, save gates PASS at checkpoint-250 and checkpoint-500 |

Held-out eval (`eval_heldout_loss.py`, 1,000 test rows, 887,947 assistant tokens scored, 0 dropped,
~1,100 s per pass on 2 GPUs):

| checkpoint | mean NLL (nats) | perplexity | delta |
|---|---|---|---|
| base `Qwen/Qwen2.5-7B-Instruct` | 1.2467 | 3.48 | — |
| `runs/qwen2.5-7b-instruct_lora_4k/final` | 0.9244 | 2.52 | **-0.32 nats** |

9.63 M input tokens were seen over the 500 steps (global batch 16 × 4096 max context ceiling).
Stated once, clearly: **a 500-step run is a plumbing check and a teaching tool, not a quality
claim** — same spirit as the [VLM tutorial](../vlm-2xH200/) and the [speech tutorial](../speech-2xH200/).

## Prerequisites

Environment setup (driver/CUDA, package set, slow-index wheel trick, stale-package cleanup) is
documented once in [../A100_2GPU.md](../A100_2GPU.md) — do that first. The stack this tutorial was
measured on: Python 3.12, torch 2.13.0+cu129, transformers 5.18.0, peft 0.21.2, accelerate 1.15.0,
datasets 5.1.0, liger-kernel 0.8.4 (installed but not used here — see §5).

Also before you start:

- **Two free GPUs.** This is a shared node with no scheduler (FoundationScale profile
  `local-single-node`, scheduler `local`). Check
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and pick two idle GPU ids;
  the launcher wrapper aborts if either target GPU holds >500 MiB of someone else's job.
- **No FP8/FlashAttention-3 on A100 (sm80).** The example runs `--attn-implementation sdpa`.
  A matmul sanity probe (**MEASURED**): warm bf16 8192³ = 257 TFLOPS on one A100, 82% of the
  312 TFLOPS datasheet dense bf16 peak. FoundationScale ships **no default device peak** — set
  `FS_DEVICE_PEAK_TFLOPS=312` `FS_DEVICE_PEAK_SOURCE="NVIDIA A100-SXM4-80GB datasheet, dense bf16"`
  if you want MFU.
- **Peak GB/GPU semantics** here = the single highest 5-second `nvidia-smi` sample during the run,
  same as the [speech-2xH200](../speech-2xH200/) and [vlm-2xH200](../vlm-2xH200/) tutorials.

All commands below run **from the repository root**, with repo-relative paths. If `$FS_ROOT` is not
your current directory, `cd` there first.

## 1. Download

Model weights and the dataset snapshot (Hub data has parquet shards under `data/`):

```bash
hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct

hf download HuggingFaceH4/ultrachat_200k --repo-type dataset \
  --include 'data/train_sft-00000-*' 'data/test_sft-*' \
  --local-dir data/ultrachat_200k
```

Only the first train shard (`train_sft-00000-*`) is needed — the recipe takes its first 20,000
conversations (see §2). Put `HF_HOME`, `PIP_CACHE_DIR`, `TMPDIR` on a big filesystem (see
[../A100_2GPU.md](../A100_2GPU.md)).

**Check it worked:** `ls models/Qwen2.5-7B-Instruct/config.json data/ultrachat_200k/data/` shows the
model config and the `train_sft-00000-*.parquet` / `test_sft-*.parquet` shards.

## 2. Data prep

Two steps. First `scripts/prepare_ultrachat.py` converts the parquet shards to ShareGPT-style rows
(`{"id", "conversations": [{"from": "human"|"gpt", "value"}]}`) — system turns dropped, conversations
with fewer than two turns skipped, **nothing truncated** (overlong rows are dropped later, at train
time, via `FOUNDATIONSCALE_TRAIN_OVERLONG=drop` set in the launcher):

```bash
python examples/llm-2xA100/scripts/prepare_ultrachat.py \
  --input-dir data/ultrachat_200k --out-dir data/llm
```

This writes `data/llm/ultrachat_raw_train.jsonl` (first 20,000 train conversations) and
`data/llm/ultrachat_raw_test.jsonl` (first 1,000 test conversations) and prints
`{"train_rows": 20000, "test_rows": 1000}`.

Second, run the *text* mode of the shared data pipeline/validator from the [VLM tutorial](../vlm-2xH200/)
over both files — same validation contract FoundationScale's conversation collator trains on:

```bash
python examples/vlm-2xH200/data/prepare_data.py text \
  data/llm/ultrachat_raw_train.jsonl data/llm/ultrachat_train.jsonl

python examples/vlm-2xH200/data/prepare_data.py text \
  data/llm/ultrachat_raw_test.jsonl data/llm/ultrachat_test.jsonl
```

**Check it worked** — expected row counts (**MEASURED**):

```bash
wc -l data/llm/ultrachat_train.jsonl data/llm/ultrachat_test.jsonl
```

| file | rows | in -> out |
|---|---|---|
| `data/llm/ultrachat_train.jsonl` | 19,996 | 20,000 raw -> 19,996 kept (4 rows: `empty_value`) |
| `data/llm/ultrachat_test.jsonl` | 1,000 | 1,000 raw -> 1,000 kept (0 dropped) |

## 3. Train

Pick two free GPU ids and launch. `STEPS` and `SAVE_EVERY` are shell overrides over the config file
(`examples/llm-2xA100/configs/qwen2.5-7b-instruct_lora_4k.env` sets `MODEL`, `CONTEXT`, `DATASET`,
`OUTPUT_DIR`, `ADAPTER_TARGETS`):

```bash
CUDA_VISIBLE_DEVICES=<two free gpu ids> STEPS=500 SAVE_EVERY=250 \
  bash examples/llm-2xA100/scripts/train_llm.sh \
  examples/llm-2xA100/configs/qwen2.5-7b-instruct_lora_4k.env
```

The launcher (`examples/llm-2xA100/scripts/train_llm.sh`) is the whole recipe: FSDP over 2 GPUs,
bf16, sdpa, gradient checkpointing, per-device batch 1 × `GRAD_ACCUM` 8 × 2 GPUs = global 16, LoRA
r16/alpha32, 4096 context, and two environment declarations FoundationScale never guesses for you —
`FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN=conversations` and
`FOUNDATIONSCALE_TRAIN_OVERLONG=drop`. Its defaults are `STEPS=1000`, `SAVE_EVERY=200`, `LR=1e-4`,
`RANK=16`, `ALPHA=32`, `GRAD_ACCUM=8`; any of them can be overridden from the shell.

**What PASS looks like** (**MEASURED**): torchrun exits **0**; FoundationScale logs
`[fs:train:done] PASS`; the save gates report **PASS** at checkpoint-250 and checkpoint-500. Measured
totals at `STEPS=500`: `train_runtime` 2341 s (4.68 s/step), `train_loss` 0.9208, peak 21.1 GB/GPU,
~4,100 train tokens/s, 9.63 M input tokens seen. Output lands in
`runs/qwen2.5-7b-instruct_lora_4k/`, adapter in `runs/qwen2.5-7b-instruct_lora_4k/final`.

If `torchrun` reports a bare `exit 1`, read FoundationScale's real verdict code on the
`exitcode : N` line inside torchrun's own per-rank failure report, not torchrun's wrapper exit (same
trap as [speech-2xH200](../speech-2xH200/)).

## 4. Evaluate

Base pass and adapter pass, both in held-out NLL of the assistant turns over `data/llm/ultrachat_test.jsonl`.
`eval_heldout_loss.py` lives with the [VLM tutorial](../vlm-2xH200/) (`examples/vlm-2xH200/scripts/`) and
loads an `AutoModelForCausalLM` when the model config has no `vision_config` — that is how a text-only
model goes through the same script (**MEASURED**):

```bash
python examples/vlm-2xH200/scripts/eval_heldout_loss.py \
  --model models/Qwen2.5-7B-Instruct \
  --dataset data/llm/ultrachat_test.jsonl \
  --image-column none --max-length 4096

python examples/vlm-2xH200/scripts/eval_heldout_loss.py \
  --model models/Qwen2.5-7B-Instruct \
  --adapter runs/qwen2.5-7b-instruct_lora_4k/final \
  --dataset data/llm/ultrachat_test.jsonl \
  --image-column none --max-length 4096
```

`--image-column none` and `--max-length 4096` are not optional here: text-only data has no image
column, and the max length must match the training context so the same rows score.

**Expected numbers** (**MEASURED**, 1,000 test rows, 887,947 assistant tokens, 0 dropped, ~1,100 s per
pass on 2 GPUs):

| run | mean NLL (nats) | ppl |
|---|---|---|
| base | 1.2467 | 3.48 |
| adapter `runs/qwen2.5-7b-instruct_lora_4k/final` | 0.9244 | 2.52 |

Delta **-0.32 nats**. Again: in-distribution UltraChat-on-UltraChat. It shows the trainer, the
adapter and the eval path all work — nothing about general chat quality.

## 5. Adapting to another text LLM

Three settings decide whether a new text model trains; the first two tripped this example.

- **`ADAPTER_TARGETS`** — required for model families FoundationScale has no `FamilySpec` for
  (`qwen2`, `llama`, ...). Name the LoRA target modules explicitly in the config as a
  space-separated list; the launcher turns each into a repeated `--adapter-target`. `qwen2` needed
  `ADAPTER_TARGETS="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"` (see
  `examples/llm-2xA100/configs/qwen2.5-7b-instruct_lora_4k.env`). For families a `FamilySpec` does
  claim, targets resolve automatically and you can leave it empty (**UNMEASURED** for such families
  in this text-only path).
- **`FUSED_LOSS`** — opt-in, off by default. The launcher passes `--fused-loss "$FUSED_LOSS"` only
  when `FUSED_LOSS` is set; e.g. `FUSED_LOSS=liger`. Do not assume it is available.
- **Config shims for new data** — if your JSONL uses different column names, declare them (this
  example sets `FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN=conversations`) and choose an overlong
  policy (`FOUNDATIONSCALE_TRAIN_OVERLONG=drop` drops rows beyond the context; this recipe never
  truncates).

Refusals met on this example's way, each by design. If a new model refuses the same way, the table
is the fix:

| refusal (exit 96, message summary) | exact cause | fix |
|---|---|---|
| `--fused-loss liger` REFUSED for `model_type` `qwen2` | the liger registration covers the `gemma4` / `qwen3_5` families only; `qwen2` is not in it | make the flag opt-in via `FUSED_LOSS` (the launcher already does), or run without fused loss — that is the measured recipe |
| adapter `lora` with no targets REFUSED (no modules resolved) | no `FamilySpec` claims `qwen2`, so nothing auto-scopes LoRA targets | set `ADAPTER_TARGETS` in the config; the launcher passes each as `--adapter-target` |

## 6. FoundationScale changes this example needed

Brief — text-only LLM support is new work, not just example files (unit tests in
`tests/train/test_conversation_text_tokenizer.py`). Full detail in the code comments and tests.

1. **`train/conversation.py`** — text-only LM support: (a) content-part *lists* are flattened to a
   plain string before `apply_chat_template` for bare tokenizers (Qwen2.5's template does
   `'<|im_start|>' + role + '\n' + content` and raised `TypeError` on a list); (b) **no** dummy-image
   injection for bare tokenizers (there is no media tower; the injected image reached the text
   template); (c) a `_text_tokenizer` accessor replaces `processor.tokenizer` (a bare tokenizer has
   no `.tokenizer`). Multimodal processors are unchanged (identity).
2. **`examples/vlm-2xH200/scripts/eval_heldout_loss.py`** — loads `AutoModelForCausalLM` when the
   config has no `vision_config`, and records `model_class`.
3. **`examples/vlm-2xH200/scripts/train_sft.sh`** — `VIDEO_COLUMN=none` skips the video column
   declaration (image/text-only data was REFUSED because a declared `video` column was absent).
   The default is unchanged.
