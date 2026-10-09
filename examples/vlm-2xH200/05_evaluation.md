# 05 — Evaluation: held-out loss, benchmarks and sample generations

In this chapter you evaluate the adapter you trained in chapter 04 on data that was never seen during training, merge the adapter into a single checkpoint for serving, and score both the base model and the fine-tuned model on public benchmarks with vLLM. Always compare **base vs fine-tuned on the same held-out rows and the same benchmark rows**, and save the JSON results — they are your evidence.

(The measured numbers below were produced on the 2x H200 VM with `gemma-4-12B-it`. Measured timings for this chapter: ~20 min of evaluation, plus a 6 min one-off vLLM install.)

## 1. Confirm your held-out split is clean

The held-out file is built in chapter 02 from the three per-source eval splits (`cat data/prepared/image.eval.jsonl data/prepared/video.eval.jsonl data/prepared/text.eval.jsonl > data/prepared/heldout.jsonl`), so evaluation rows never appear in training. Locate the file:

```bash
ls -l data/prepared/heldout.jsonl
```

**Check it worked.** The file exists and is non-empty. Evaluation rows must not overlap training rows — if you rebuilt the data yourself, re-run `prepare_data.py split` from chapter 02 and then the `cat` command above, rather than hand-splitting; the measured train/held-out split is exactly this (a train split made with `prepare_data.py split`), and its held-out file holds **171 disjoint rows: 110 image, 40 video, 21 text**.

## 2. Held-out loss (NLL and perplexity)

Run the held-out loss script. It uses FoundationScale's own collator, so **the same tokens are supervised as during training**, which makes the number directly comparable across runs.

`eval_heldout_loss.py` takes `--video-frames N`, the number of frames sampled per video clip. Pass the **same value as `FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES`** used during training (16 frames/clip in the measured runs). Without `--video-frames`, video rows are refused and the script exits with code 96.

Evaluate the base model:

```bash
python examples/vlm-2xH200/scripts/eval_heldout_loss.py \
  --model models/gemma-4-12B-it \
  --data data/prepared/heldout.jsonl \
  --max-length 16384 \
  --video-frames 16 \
  --out results/heldout_base.json
```

Evaluate a checkpoint:

```bash
python examples/vlm-2xH200/scripts/eval_heldout_loss.py \
  --model models/gemma-4-12B-it \
  --adapter runs/<run>/checkpoint-N \
  --data data/prepared/heldout.jsonl \
  --max-length 16384 \
  --video-frames 16 \
  --out results/heldout_lora.json
```

Both commands use both GPUs via `device_map=auto`. The output block reports `rows used`, `rows dropped`, `mean NLL ... nats/token`, and `perplexity`. Rows that do not fit `--max-length` are dropped and counted.

**Check it worked.** Measured on 30 rows (script check; note that in that particular check the rows partly overlapped training — your chapter 02 held-out split does not):

| model | mean NLL (nats/token) | perplexity | time |
|---|---|---|---|
| base `gemma-4-12B-it` | 1.601 | 4.96 | ~90 s |
| + 6-step LoRA adapter | 1.149 | 3.16 | ~90 s |

A 6-step adapter is a **plumbing check, not a quality claim** — the above numbers only prove the pipeline runs end to end. Train for hundreds of steps before comparing.

After a real **300-step** run (`gemma-4-12B-it`, 16K, 300 steps, grad accumulation 4, unpadded; LoRA r16/a32 on 2x H200 with FSDP, bf16, activation checkpointing and `--fused-loss liger`, on a train split made with `prepare_data.py split`), the training loss fell **1.11 → 0.43** (exit 0) and the measured held-out result (171 disjoint rows: 110 image, 40 video, 21 text) is:

| model | mean NLL (nats/token) | perplexity | rows used | rows dropped |
|---|---|---|---|---|
| base `gemma-4-12B-it` | 1.433 | 4.19 | 171 | 0 |
| + 300-step LoRA adapter | 0.505 | 1.66 | 171 | 0 |

This is the real quality signal: fine-tuning cut the held-out NLL from **1.433 to 0.505** (perplexity **4.19 → 1.66**) with **0 rows dropped**.

For each JSON you write, record `rows used` and `rows dropped` alongside the loss so runs with different drop rates are not compared naively.

## 3. Merge the adapter for serving

Benchmarks and serving run more conveniently from one merged checkpoint. Merge the LoRA adapter into the base weights on CPU (a few minutes):

```bash
python examples/vlm-2xH200/scripts/merge_lora.py \
  --base models/gemma-4-12B-it \
  --adapter runs/<run>/checkpoint-N \
  --out merged/gemma-4-12B-it-lora
```

**Check it worked.** The script prints **8 `PASS` check lines**. They cover at least these three kinds of check:

- reloaded params equal base params,
- no LoRA weights left in memory or on disk,
- processor saved and reloadable.

Look for all 8 `PASS` lines in the output before continuing.

## 4. Benchmark accuracy with vLLM

> [!WARNING]
> **A benchmark number is only as good as its answer parser. A scoring bug once reported the same base model at 47.00% on MMStar instead of 64.02% — 17 points too low.** An earlier version of `eval_mcq_vllm.py` generated with `max_tokens=32` and took the **first standalone capital letter** as the answer. Two failure modes followed: reasoning answers were cut off mid-thought (**26% unparsable**), and the English article "A" inside an answer sentence was read as option A. The script now defaults to `--max-tokens 512` and prefers an explicit answer (`Answer: C`, `the answer is (C)`, `**C**`), then a bare letter, then the last letter on the final line. **Always check the `unparsable` count and read a few raw generated outputs before you trust a benchmark number**, and raise `--max-tokens` to 1024 for reasoning-heavy benchmarks.

### 4a. Create a separate vLLM environment

vLLM pins its own stack, so install it into its own virtualenv, not the training environment (the separate venv keeps vLLM from changing the training env):

```bash
python -m venv vllm-env && vllm-env/bin/pip install vllm
```

Tested with **vLLM 0.30–0.31 with torch 2.13** (the measured stack was **vLLM 0.30.0 + torch 2.13 + transformers 5.18**; no explicit version pin is required). The install takes about **6 min**.

```bash
vllm-env/bin/python -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)"
```

**Check it worked.** Version prints close to the measured stack (vLLM 0.30.0, torch 2.13, transformers 5.18).

### 4b. Get the benchmark TSVs

Download the VLMEvalKit TSVs (`MMStar`, `MMBench_DEV_EN`, `AI2D_TEST`) from `https://opencompass.openxlab.space/utils/VLMEval/<NAME>.tsv`. For MMStar:

```bash
mkdir -p data/eval
curl -L -o data/eval/MMStar.tsv https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv
```

**If it fails with "certificate has expired"** (curl error 60): the opencompass host's TLS certificate was expired when tested. Retry with `curl -k` — the file is public data:

```bash
curl -k -L -o data/eval/MMStar.tsv https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv
```

Check the row count looks right:

```bash
wc -l data/eval/MMStar.tsv
```

**Check it worked.** `wc -l` prints ~1,500 rows. Expect **~1,500 questions**; the scorer skips 2 malformed rows, so the measured `MMStar` TSV scores **1,498 questions**.

The columns are: `index`, `question`, `A`..`D`, `answer`, `image` (base64), optional `hint`, `category`. Store them under `data/eval/`, e.g. `data/eval/MMStar.tsv`. (The same URL pattern with a different `<NAME>` gives `MMBench_DEV_EN.tsv` and `AI2D_TEST.tsv`.)

### 4c. Run a benchmark

**Important VM facts:** this VM has **no CUDA toolkit (`nvcc`)**, so you must:

1. set `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_USE_DEEP_GEMM=0`, and
2. use `--tp 1` — tensor parallel 2 triggers FlashInfer's fused all-reduce, which JIT-compiles with `nvcc` and fails.

Every target model fits one H200 for inference, so run **two benchmarks in parallel, one per GPU** (`CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`). **Compare base vs fine-tuned on the SAME benchmark rows** — both runs below score `data/eval/MMStar.tsv`, so the two numbers are directly comparable.

Scoring controls (see the warning above):

- `--max-tokens N` — maximum number of new tokens generated per question. The script defaults to **512**; the old fixed `max_tokens=32` cut reasoning answers off (26% unparsable) and made numbers untrustworthy. Raise to **1024** for reasoning-heavy benchmarks.
- `--limit N` — score only the first N questions. Fine for a smoke test, but score the **same** rows for every model you compare, or drop `--limit` entirely (the measured numbers below are the full 1,498-question MMStar set).

GPU 0 (merged adapter):

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 \
  vllm-env/bin/python examples/vlm-2xH200/scripts/eval_mcq_vllm.py \
  --model merged/gemma-4-12B-it-lora \
  --tsv data/eval/MMStar.tsv \
  --name MMStar \
  --limit 100 \
  --max-tokens 512 \
  --tp 1 \
  --max-model-len 8192 \
  --out results/mmstar_lora.json
```

GPU 1 (base model, **the same MMStar rows** for comparison):

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 \
  vllm-env/bin/python examples/vlm-2xH200/scripts/eval_mcq_vllm.py \
  --model models/gemma-4-12B-it \
  --tsv data/eval/MMStar.tsv \
  --name MMStar \
  --limit 100 \
  --max-tokens 512 \
  --tp 1 \
  --max-model-len 8192 \
  --out results/mmstar_base.json
```

For Qwen3.6, add `--disable-thinking` so it answers with a letter directly.

**Check it worked.** Measured on MMStar, the full 1,498 questions, `--max-tokens 512`, base vs the merged 300-step adapter (trained on Vietnamese documents and activity videos):

| model | accuracy | unparsable |
|---|---|---|
| base `gemma-4-12B-it` | 64.02% | 112 |
| + 300-step LoRA adapter (merged) | 61.88% | 105 |

Specialising on Vietnamese documents and activity videos **cost about 2 points of general English perception** (64.02% → 61.88%) — a real trade-off you only see when base and fine-tuned are scored on the same rows. Read that against the held-out numbers in section 2: the adapter is much better on its own domain while giving up a little general English perception.

Again: a 6-step run is a plumbing check, not a quality claim. Note also that the **first 20 questions alone score 20% (both runs, identical predictions)** — a small slice of a benchmark is not representative, so **always report the full set** (or at least the same `--limit` for every model you compare). And per the warning above, check the `unparsable` counts (112 base / 105 fine-tuned here) and read a few raw outputs before quoting these numbers.

Save every result JSON (`results/*.json`). Report base vs fine-tuned on the same held-out rows and the same benchmark rows.

## 5. Look at sample generations

Metrics hide failure modes; read a handful of generations from held-out rows.

TODO(verify) command — use `vllm chat` (from the `vllm-env` environment) on a few rows of `data/prepared/heldout.jsonl` TODO(verify) exact invocation, including how images from the rows are passed to the chat CLI.

**Check it worked.** TODO(verify) expected output — the responses are coherent answers to the held-out prompts and conform to the training answer format.

## Troubleshooting

- **`nvcc` / FlashInfer JIT compile failure at startup, or with `--tp 2`.** The VM has no CUDA toolkit. Set `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_USE_DEEP_GEMM=0` and run with `--tp 1`. Tensor parallel 2 triggers FlashInfer's fused all-reduce, which JIT-compiles with `nvcc` and fails on this VM.
- **vLLM import conflicts / wrong torch version.** vLLM pins its own stack. Run benchmarks only from `vllm-env` (`vllm-env/bin/python`), never from the training environment.
- **Benchmark TSV download fails with "certificate has expired" (curl error 60).** The opencompass host's TLS certificate can be expired. Retry with `curl -k` (the file is public data) and check with `wc -l` that you have ~1,500 rows.
- **Wrong or tiny TSV.** Check `wc -l data/eval/<name>.tsv`: MMStar has ~1,500 rows (1,498 usable — the scorer skips 2 malformed rows). Re-download if the file is truncated or empty.
- **`merge_lora.py` does not print all 8 `PASS` check lines.** Do not use the merged checkpoint. Re-check that `--adapter` points to a complete `checkpoint-N` from chapter 04 and re-run the merge.
- **`eval_heldout_loss.py` refuses video rows with `exit 96`.** Video rows only run with `--video-frames N`. Pass the same value as `FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES` from training (16 frames/clip in the measured runs).
- **Held-out loss looks suspiciously good.** Check "rows used" / "rows dropped" and verify your held-out rows never appeared in training. A script check that let rows partly overlap training measured NLL 1.601 → 1.149 in only 6 steps; that kind of gap can be an artifact of overlap rather than learning. (For scale: on the clean 171-row split the 300-step adapter measured 1.433 → 0.505 with 0 rows dropped.)
- **Benchmark score varies wildly between runs or reports.** Do not extrapolate from a subset: the first 20 MMStar questions scored 20% while the first 100 scored 54%. Use the full set (or an identical `--limit` for all models). If a score looks far too **low**, read the warning in section 4 first: check the `unparsable` count and a few raw outputs (a 32-token budget plus first-capital-letter parsing once reported 47.00% where the same base model scores 64.02%).
- **Model does not answer with a multiple-choice letter (Qwen3.6).** Add `--disable-thinking` so it answers with a letter directly.
- **Base and fine-tuned numbers are not comparable.** They were probably scored on different TSVs or different `--limit` slices. Score both on the same benchmark file (both `data/eval/MMStar.tsv` here) and the same rows, one run per GPU.
- **Out of memory during `eval_heldout_loss.py` at `--max-length 16384`.** Not expected: the measured 171-row held-out eval ran at `--max-length 16384` to completion with 0 rows dropped. If you still hit OOM (a larger model or longer rows), lower `--max-length` and account for the increased `rows dropped` count.

Next: repeated-training runs at scale (multi-checkpoint sweeps, hundreds of steps) and qualitative error analysis on the saved JSONs.

**Next:** [06 — Ready-to-run configs for each model](README.md#06--ready-to-run-configs-for-each-model)