# 05 — Evaluation: held-out loss and benchmarks

Evaluate the adapter trained in chapter 04 on data never seen during training, merge it into a single checkpoint for serving, and score both the base model and the fine-tuned model on public benchmarks with vLLM. Always compare **base vs fine-tuned on the same held-out rows and the same benchmark rows**, and save the JSON results — they are your evidence.

Every command below assumes an active environment (`source .venv/bin/activate`); the vLLM commands of section 4 run from their own `vllm-env`. The measured numbers come from `gemma-4-12B-it` on a 2x H200 setup.

## 1. Confirm your held-out split is clean

The held-out file is built in chapter 02 from the three per-source eval splits (`cat data/prepared/image.eval.jsonl data/prepared/video.eval.jsonl data/prepared/text.eval.jsonl > data/prepared/heldout.jsonl`), so evaluation rows never appear in training. Locate the file:

```bash
ls -l data/prepared/heldout.jsonl
```

**Check it worked.** The file exists and is non-empty. Evaluation rows must not overlap training rows — if you rebuilt the data yourself, re-run `prepare_data.py split` from chapter 02 and then the `cat` command above, rather than hand-splitting. The measured train/held-out split is exactly this, and its held-out file holds **171 disjoint rows: 110 image, 40 video, 21 text**.

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

**Check it worked.** Measured on 30 rows (a script check whose rows partly overlapped training — your chapter 02 held-out split does not):

| model | mean NLL (nats/token) | perplexity | time |
|---|---|---|---|
| base `gemma-4-12B-it` | 1.601 | 4.96 | ~90 s |
| + 6-step LoRA adapter | 1.149 | 3.16 | ~90 s |

A 6-step adapter is a **plumbing check, not a quality claim** — the numbers only prove the pipeline runs end to end. Train for hundreds of steps before comparing.

After a real **300-step** run (`gemma-4-12B-it`, 16K, 300 steps, grad accumulation 4, unpadded; LoRA r16/a32 on 2x H200 with FSDP, bf16, activation checkpointing and `--fused-loss liger`, on a train split made with `prepare_data.py split`), the training loss fell **1.11 → 0.43** (exit 0) and the measured held-out result (171 disjoint rows: 110 image, 40 video, 21 text) is:

| model | mean NLL (nats/token) | perplexity | rows used | rows dropped |
|---|---|---|---|---|
| base `gemma-4-12B-it` | 1.433 | 4.19 | 171 | 0 |
| + 300-step LoRA adapter | 0.505 | 1.66 | 171 | 0 |

The clean quality signal: fine-tuning cut the held-out NLL from **1.433 to 0.505** (perplexity **4.19 → 1.66**) with **0 rows dropped**.

For each JSON you write, record `rows used` and `rows dropped` alongside the loss so runs with different drop rates are not compared naively.

## 3. Merge the adapter for serving

Benchmarks and serving run more conveniently from one merged checkpoint. Where the adapter lives depends on the trainer you used in chapter 04:

- `train_sft.sh` saves adapter checkpoints as `runs/<run>/checkpoint-N/` — use the `checkpoint-N` you want to serve;
- `train_dpo.py` and `train_grpo.py` save the final adapter as `runs/<run>/final/`.

Merge the LoRA adapter into the base weights on CPU (a few minutes). For an SFT run (`train_sft.sh`), point `--adapter` at a `checkpoint-N`:

```bash
python examples/vlm-2xH200/scripts/merge_lora.py \
  --base models/gemma-4-12B-it \
  --adapter runs/<run>/checkpoint-N \
  --out merged/gemma-4-12B-it-lora
```

For a `train_dpo.py` or `train_grpo.py` run the only difference is the adapter path (`runs/<run>/final`):

```bash
python examples/vlm-2xH200/scripts/merge_lora.py \
  --base models/gemma-4-12B-it \
  --adapter runs/<run>/final \
  --out merged/gemma-4-12B-it-lora
```

**Check it worked.** The script prints **8 `PASS` check lines**. They cover at least these three kinds of check:

- reloaded params equal base params,
- no LoRA weights left in memory or on disk,
- processor saved and reloadable.

Look for all 8 `PASS` lines in the output before continuing.

## 4. Benchmark accuracy with vLLM

> [!WARNING]
> **A benchmark number is only as good as its answer parser.** A parser bug once reported the same base model at 47.00% on MMStar instead of 64.02% — 17 points too low: `max_tokens=32` cut reasoning answers off (26% unparsable) and first-capital-letter parsing read English articles as option letters. `eval_mcq_vllm.py` now defaults to `--max-tokens 512` and prefers an explicit answer (`Answer: C`, `the answer is (C)`, `**C**`), then a bare letter, then the last letter on the final line. **Always check the `unparsable` count and read a few raw generated outputs before you trust a benchmark number**, and raise `--max-tokens` to 1024 for reasoning-heavy benchmarks.

### 4a. Create a separate vLLM environment

vLLM pins its own stack, so install it into its own virtualenv, not the training environment (the separate venv keeps vLLM from changing the training env):

```bash
python -m venv vllm-env && vllm-env/bin/pip install vllm
```

Tested with **vLLM 0.30–0.31 with torch 2.13** (the measured stack was **vLLM 0.30.0 + torch 2.13 + transformers 5.18**; no explicit version pin is required).

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

**If it fails with "certificate has expired"** (curl error 60): the opencompass host's TLS certificate has been known to expire. Retry with `curl -k` — the file is public data:

```bash
curl -k -L -o data/eval/MMStar.tsv https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv
```

**Do not check the download with `wc -l`.** Fields such as `question` contain embedded newlines (the MMStar TSV is 2,107 *lines* but only 1,500 *rows*), so line count massively over-counts. Count rows with Python's TSV reader instead:

```bash
python -c "import csv,sys;csv.field_size_limit(sys.maxsize);print(sum(1 for _ in csv.DictReader(open('data/eval/MMStar.tsv'),delimiter='\t')))"
```

**Check it worked.** Expect about **1,498–1,500 questions** from the Python one-liner above or the scorer's `questions : N (skipped rows: M)` line. The scorer skips 2 malformed rows, so the measured `MMStar` TSV scores **1,498 questions**.

The columns are: `index`, `question`, `A`..`D`, `answer`, `image` (base64), optional `hint`, `category`. Store them under `data/eval/`, e.g. `data/eval/MMStar.tsv`. (The same URL pattern with a different `<NAME>` gives `MMBench_DEV_EN.tsv` and `AI2D_TEST.tsv`.)

### 4c. Run a benchmark

On a machine without the CUDA toolkit (`nvcc`), you must:

1. set `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_USE_DEEP_GEMM=0`, and
2. use `--tp 1` — tensor parallel 2 triggers FlashInfer's fused all-reduce, which JIT-compiles with `nvcc` and fails.

Every target model fits one H200 for inference, so run **two benchmarks in parallel, one per GPU** (`CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`). **Compare base vs fine-tuned on the SAME benchmark rows** — both runs below score `data/eval/MMStar.tsv`, so the two numbers are directly comparable.

Scoring controls:

- `--max-tokens N` — maximum number of new tokens generated per question. The script defaults to **512** (the old fixed `max_tokens=32` caused 26% unparsable). Raise to **1024** for reasoning-heavy benchmarks.
- `--limit N` — score only the first N questions. Fine for a smoke test, but score the **same** rows for every model you compare, or drop `--limit` entirely (the measured numbers below are the full 1,498-question MMStar set).
- `--max-num-seqs N` — maximum number of concurrent sequences vLLM runs. The script defaults to **256**. vLLM's own default of **1,024** concurrent sequences exceeded the **844** linear-attention cache blocks that Qwen3.6-27B gets on one H200, and the engine refused to start — keep the default, or lower it further if vLLM reports `max_num_seqs exceeds available Mamba cache blocks`.

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

Again, a 6-step run is a plumbing check, not a quality claim. A small slice of a benchmark is not representative: the first 20 MMStar questions alone score 20% and the first 100 scored 54% (both runs, identical predictions), so **always report the full set** (or at least the same `--limit` for every model you compare). Check the `unparsable` counts (112 base / 105 fine-tuned here) and read a few raw outputs before quoting these numbers.

Save every result JSON (`results/*.json`). Report base vs fine-tuned on the same held-out rows and the same benchmark rows.

## Troubleshooting

- **`nvcc` / FlashInfer JIT compile failure at startup, or with `--tp 2`.** Without a CUDA toolkit installed, set `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_USE_DEEP_GEMM=0` and run with `--tp 1`. Tensor parallel 2 triggers FlashInfer's fused all-reduce, which JIT-compiles with `nvcc`.
- **vLLM import conflicts / wrong torch version.** vLLM pins its own stack. Run benchmarks only from `vllm-env` (`vllm-env/bin/python`), never from the training environment.
- **vLLM refuses to start: `max_num_seqs exceeds available Mamba cache blocks`.** Lower `--max-num-seqs` in `eval_mcq_vllm.py` (the script defaults to 256). vLLM's default of 1,024 concurrent sequences exceeded the 844 linear-attention cache blocks that Qwen3.6-27B gets on one H200, and the engine refused to start.
- **TSV download fails with "certificate has expired" (curl error 60), or the downloaded TSV is truncated or wrong.** The opencompass host's TLS certificate can be expired — retry with `curl -k` (the file is public data). Then verify you have ~1,500 **questions** with the Python TSV counter from section 4b or the scorer's `questions : N (skipped rows: M)` line, **not** `wc -l` (fields contain embedded newlines; MMStar is 2,107 lines for 1,500 rows). Re-download if the file is empty or short.
- **`merge_lora.py` does not print all 8 `PASS` check lines.** Do not use the merged checkpoint. Re-check that `--adapter` points to a complete `checkpoint-N` from chapter 04 (SFT runs; `train_dpo.py`/`train_grpo.py` runs keep the final adapter in `runs/<run>/final/`) and re-run the merge.
- **`eval_heldout_loss.py` refuses video rows with `exit 96`.** Video rows only run with `--video-frames N`. Pass the same value as `FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES` from training (16 frames/clip in the measured runs).
- **Held-out loss looks suspiciously good.** Check `rows used` / `rows dropped` and verify your held-out rows never appeared in training. A script check that let rows partly overlap training measured NLL 1.601 → 1.149 in only 6 steps; that kind of gap can be an artifact of overlap rather than learning. (For scale: on the clean 171-row split the 300-step adapter measured 1.433 → 0.505 with 0 rows dropped.)
- **Benchmark score varies wildly between runs or reports, or looks far too low.** Do not extrapolate from a subset (the first 20 MMStar questions scored 20%, the first 100 scored 54%). Use the full set (or an identical `--limit` for all models). If a score looks far too **low**, check the `unparsable` count and a few raw outputs first (a 32-token budget plus first-capital-letter parsing once reported 47.00% where the same base model scores 64.02%).
- **Model does not answer with a multiple-choice letter (Qwen3.6).** Add `--disable-thinking` so it answers with a letter directly.
- **Base and fine-tuned numbers are not comparable.** They were probably scored on different TSVs or different `--limit` slices. Score both on the same benchmark file (both `data/eval/MMStar.tsv` here) and the same rows, one run per GPU.
- **Out of memory during `eval_heldout_loss.py` at `--max-length 16384`.** Not expected: the measured 171-row held-out eval ran at `--max-length 16384` to completion with 0 rows dropped. If you still hit OOM (a larger model or longer rows), lower `--max-length` and account for the increased `rows dropped` count.

Next: repeated-training runs at scale (multi-checkpoint sweeps, hundreds of steps) and qualitative error analysis on the saved JSONs.

**Next:** [06 — Ready-to-run configs for each model](README.md#06--ready-to-run-configs-for-each-model)
