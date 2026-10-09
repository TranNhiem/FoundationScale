# 05 — Evaluation: held-out loss, benchmarks and sample generations

In this chapter you evaluate the adapter you trained in chapter 04 on data that was never seen during training, merge the adapter into a single checkpoint for serving, and score both the base model and the fine-tuned model on public benchmarks with vLLM. Always compare **base vs fine-tuned on the same held-out rows and the same benchmark rows**, and save the JSON results — they are your evidence.

(The measured numbers below were produced on the 2x H200 VM with `gemma-4-12B-it`.)

## 1. Confirm your held-out split is clean

The held-out split comes from the `prepare_data.py` split from chapter 02, so evaluation rows never appear in training. Locate the file:

```bash
ls -l data/prepared/heldout.jsonl
```

**Check it worked.** The file exists and is non-empty. Evaluation rows must not overlap training rows — if you rebuilt the data yourself, re-run `prepare_data.py` from chapter 02 rather than hand-splitting TODO(verify) the exact split flags.

## 2. Held-out loss (NLL and perplexity)

Run the held-out loss script. It uses FoundationScale's own collator, so **the same tokens are supervised as during training**, which makes the number directly comparable across runs.

Evaluate the base model:

```bash
python examples/vlm-2xH200/scripts/eval_heldout_loss.py \
  --model models/gemma-4-12B-it \
  --data data/prepared/heldout.jsonl \
  --max-length 16384 \
  --out results/heldout_base.json
```

Evaluate a checkpoint:

```bash
python examples/vlm-2xH200/scripts/eval_heldout_loss.py \
  --model models/gemma-4-12B-it \
  --adapter runs/<run>/checkpoint-N \
  --data data/prepared/heldout.jsonl \
  --max-length 16384 \
  --out results/heldout_lora.json
```

Both commands use both GPUs via `device_map=auto`. The output block reports `rows used`, `rows dropped`, `mean NLL ... nats/token`, and `perplexity`. Rows that do not fit `--max-length` are dropped and counted.

**Check it worked.** Measured on 30 rows (script check; note that in that particular check the rows partly overlapped training — your chapter 02 held-out split does not):

| model | mean NLL (nats/token) | perplexity | time |
|---|---|---|---|
| base `gemma-4-12B-it` | 1.601 | 4.96 | ~90 s |
| + 6-step LoRA adapter | 1.149 | 3.16 | ~90 s |

A 6-step adapter is a **plumbing check, not a quality claim** — the above numbers only prove the pipeline runs end to end. Train for hundreds of steps before comparing.

For each JSON you write, record `rows used` and `rows dropped` alongside the loss so runs with different drop rates are not compared naively.

## 3. Merge the adapter for serving

Benchmarks and serving run more conveniently from one merged checkpoint. Merge the LoRA adapter into the base weights on CPU (a few minutes):

```bash
python examples/vlm-2xH200/scripts/merge_lora.py \
  --base models/gemma-4-12B-it \
  --adapter runs/<run>/checkpoint-N \
  --out merged/gemma-4-12B-it-lora
```

**Check it worked.** The script prints `PASS` checks for all three of these:

- reloaded params equal base params,
- no LoRA weights left in memory or on disk,
- processor saved and reloadable.

Look for the three `PASS` lines in the output before continuing.

## 4. Benchmark accuracy with vLLM

### 4a. Create a separate vLLM environment

vLLM pins its own stack, so install it into its own virtualenv, not the training environment:

```bash
python -m vvenv vllm-env  # TODO(verify) — use: python -m venv vllm-env
```

```bash
python -m venv vllm-env && vllm-env/bin/pip install vllm
```

TODO(verify) the exact vLLM version to pin. Measured with **vLLM 0.30.0 + torch 2.13 + transformers 5.18**.

```bash
vllm-env/bin/python -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)"
```

**Check it worked.** Version prints close to the measured stack (vLLM 0.30.0, torch 2.13, transformers 5.18).

### 4b. Get the benchmark TSVs

Download the VLMEvalKit TSVs (`MMStar`, `MMBench_DEV_EN`, `AI2D_TEST`) from:

```
https://opencompass.openxlab.space/utils/VLMEval/<NAME>.tsv
```

TODO(verify) the exact URLs. The columns are: `index`, `question`, `A`..`D`, `answer`, `image` (base64), optional `hint`, `category`.

Store them under `data/eval/`, e.g. `data/eval/MMStar.tsv`.

### 4c. Run a benchmark

**Important VM facts:** this VM has **no CUDA toolkit (`nvcc`)**, so you must:

1. set `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_USE_DEEP_GEMM=0`, and
2. use `--tp 1` — tensor parallel 2 triggers FlashInfer's fused all-reduce, which JIT-compiles with `nvcc` and fails.

Every target model fits one H200 for inference, so run **two benchmarks in parallel, one per GPU** (`CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`).

GPU 0 (merged 6-step adapter):

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 \
  vllm-env/bin/python examples/vlm-2xH200/scripts/eval_mcq_vllm.py \
  --model merged/gemma-4-12B-it-lora \
  --tsv data/eval/MMStar.tsv \
  --name MMStar \
  --limit 100 \
  --tp 1 \
  --max-model-len 8192 \
  --out results/mmstar.json
```

GPU 1 (base model, same rows for comparison) — TODO(verify) that `--model models/gemma-4-12B-it` is accepted alongside `--tsv data/eval/MMBench_DEV_EN.tsv`:

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 \
  vllm-env/bin/python examples/vlm-2xH200/scripts/eval_mcq_vllm.py \
  --model models/gemma-4-12B-it \
  --tsv data/eval/MMBench_DEV_EN.tsv \
  --name MMBench_DEV_EN \
  --limit 100 \
  --tp 1 \
  --max-model-len 8192 \
  --out results/mmbench_dev_en_base.json
```

For Qwen3.6, add `--disable-thinking` so it answers with a letter directly.

**Check it worked.** Measured on MMStar, first 100 questions:

| model | accuracy | unparsable | generation time |
|---|---|---|---|
| base `gemma-4-12B-it` | 54.00% (54/100) | 0 | 5.4 s |
| merged 6-step adapter | 54.00% | 1 | — |

Again: a 6-step run is a plumbing check, not a quality claim. Note also that the **first 20 questions alone score 20% (both runs, identical predictions)** — a small slice of a benchmark is not representative, so **always report the full set** (or at least the same `--limit` for every model you compare).

Save every result JSON (`results/*.json`). Report base vs fine-tuned on the same held-out rows and the same benchmark rows.

## 5. Look at sample generations

Metrics hide failure modes; read a handful of generations from held-out rows.

TODO(verify) command — use `vllm chat` (from the `vllm-env` environment) on a few rows of `data/prepared/heldout.jsonl` TODO(verify) exact invocation, including how images from the rows are passed to the chat CLI.

**Check it worked.** TODO(verify) expected output — the responses are coherent answers to the held-out prompts and conform to the training answer format.

## Troubleshooting

- **`nvcc` / FlashInfer JIT compile failure at startup, or with `--tp 2`.** The VM has no CUDA toolkit. Set `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_USE_DEEP_GEMM=0` and run with `--tp 1`. Tensor parallel 2 triggers FlashInfer's fused all-reduce, which JIT-compiles with `nvcc` and fails on this VM.
- **vLLM import conflicts / wrong torch version.** vLLM pins its own stack. Run benchmarks only from `vllm-env` (`vllm-env/bin/python`), never from the training environment.
- **`merge_lora.py` does not print all three `PASS` checks.** Do not use the merged checkpoint. Re-check that `--adapter` points to a complete `checkpoint-N` from chapter 04 and re-run the merge.
- **Held-out loss looks suspiciously good.** Check "rows used" / "rows dropped" and verify your held-out rows never appeared in training. A script check that let rows partly overlap training measured NLL 1.601 → 1.149 in only 6 steps; that kind of gap can be an artifact of overlap rather than learning.
- **Benchmark score varies wildly between runs or reports.** Do not extrapolate from a subset: the first 20 MMStar questions scored 20% while the first 100 scored 54%. Use the full set (or an identical `--limit` for all models).
- **Model does not answer with a multiple-choice letter (Qwen3.6).** Add `--disable-thinking` so it answers with a letter directly.
- **TODO(verify): out-of-memory during `eval_heldout_loss.py` at `--max-length 16384`.** Fallback: lower `--max-length` and account for the increased `rows dropped` count.

Next: repeated-training runs at scale (multi-checkpoint sweeps, hundreds of steps) and qualitative error analysis on the saved JSONs.

**Next:** [06 — Ready-to-run configs for each model](README.md#06--ready-to-run-configs-for-each-model)
