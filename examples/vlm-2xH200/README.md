# Train your own vision-language model on 2x H200 with FoundationScale

A hands-on path from an empty VM to an evaluated, fine-tuned VLM. It is written for teams that
each get **one machine with 2x NVIDIA H200 (141 GB each)**, and for anyone who wants a worked,
measured example of LoRA training on image + video + text data.

Everything below was run on that machine. The numbers are measured, not estimated; the raw
evidence is in [`validation_campaigns/vlm_2xh200/EVIDENCE.md`](../../validation_campaigns/vlm_2xh200/EVIDENCE.md).

## What you get

| | |
|---|---|
| Models | `Qwen/Qwen3.6-27B` (dense), `Qwen/Qwen3.6-35B-A3B` (MoE), `google/gemma-4-31B-it` (dense), `google/gemma-4-26B-A4B-it` (MoE), `google/gemma-4-12B-it` (dense) — plan ~300 GB of free disk to hold all five |
| Method | LoRA (r16 / alpha 32 by default), FSDP across both GPUs, bf16, activation checkpointing |
| Context | 16K and 32K tokens |
| Data | image + text, video + text and text-only, mixed in one run |
| Algorithms | SFT (image + video + text), DPO / IPO / ORPO / SimPO / CPO (text), GRPO / Dr-GRPO / DAPO / GSPO / REINFORCE++ / RLOO (images) |

## Chapters

1. [Installation: conda environment or Docker image](01_installation.md)
2. [Data preparation: image, video and text formats, converting and mixing](02_data_preparation.md)
3. [Training configuration: model, LoRA, context length and memory](03_training_configuration.md)
4. [Training algorithms: SFT, preference training and RL](04_training_algorithms.md)
5. [Evaluation: held-out loss, benchmarks and sample generations](05_evaluation.md)
6. [Ready-to-run configs for each model](#06--ready-to-run-configs-for-each-model) (below)

## Quick start (after chapters 1 and 2)

Get the tutorial and run everything from the repository root `FoundationScale/`:

```bash
git clone https://github.com/TranNhiem/FoundationScale.git && cd FoundationScale
bash examples/vlm-2xH200/scripts/train_sft.sh examples/vlm-2xH200/configs/gemma-4-12B-it_16k.env
```


Measured by a fresh user on 2x H200: data prep < 1 min on the sample files, 20-step SFT of
gemma-4-12B-it ~12 min (add `STEPS=20` for that smoke run), DPO 5 steps 92 s, GRPO 5 steps 198 s,
evaluation ~20 min plus a 6 min vLLM install.

## Measured results: mixed image + video + text, LoRA, 2x H200

300 rows (120 image, 120 video with 16 frames per clip, 60 text), every batch padded to the
full context, so peak memory is the worst case. Peak memory is per GPU; tokens/s is the total
over both GPUs (6-step runs, so throughput is approximate).

| model | 16K peak | 16K tokens/s | 32K peak | 32K tokens/s |
|---|---|---|---|---|
| gemma-4-12B-it | 32.3 GB | 2,596 | 49.7 GB | 1,582 |
| gemma-4-26B-A4B-it (MoE) | 48.2 GB | 3,626 | 65.7 GB | 2,526 |
| gemma-4-31B-it | 63.0 GB | 1,088 | 90.2 GB | 650 |
| Qwen3.6-27B | 51.0 GB | 3,372 | 70.3 GB | 3,024 |
| Qwen3.6-35B-A3B (MoE) | 48.5 GB | 8,009 | 58.0 GB | 8,071 |

All ten runs exit 0, pass FoundationScale's save gate, and write a LoRA adapter that loads with
`PeftModel.from_pretrained`. Preference and RL training with LoRA (chapter 4): DPO on
Qwen3.6-27B peaks at 72.7 GB per GPU at 4K and 8K tokens; GRPO with images peaks at about 55 GB
(gemma-4-12B-it) and 69 GB (Qwen3.6-27B).

### Longer context: 64K and 128K (65536 / 131072 tokens)

Same setup as above: LoRA r16 / alpha 32, FSDP, bf16, activation checkpointing, `--fused-loss
liger`; mixed image + video + text with 16 frames per clip. The context runs are **padded to the
full length**, so these numbers are the worst case. Peak memory is per GPU, tokens/s is over
both GPUs.

| model | 64K peak | 64K tokens/s | 128K |
|---|---|---|---|
| gemma-4-12B-it | 90.3 GB | 861 | out of memory |
| gemma-4-26B-A4B-it (MoE) | 106.6 GB | 1,379 | out of memory |
| gemma-4-31B-it | out of memory — its maximum on 2x H200 is 32K | — | out of memory |
| Qwen3.6-27B | 113.8 GB | 2,042 | out of memory |
| Qwen3.6-35B-A3B (MoE) | 81.2 GB | 4,830 | out of memory |

- 65536 tokens: gemma-4-12B-it, gemma-4-26B-A4B-it, Qwen3.6-27B and Qwen3.6-35B-A3B all **exit
  0**. gemma-4-31B-it is **out of memory** at 64K; its maximum context on 2x H200 is 32K.
- 131072 tokens: **out of memory on all five models** with this setup. You would need context
  parallelism or CPU offload.

### Algorithms: 20-step runs (all exit 0)

| algorithm | model(s) | data |
|---|---|---|
| DPO (LoRA) | all five models | text |
| IPO, SimPO, CPO, ORPO | Qwen3.6-27B | text |
| GRPO, Dr-GRPO, DAPO, GSPO, REINFORCE++, RLOO | gemma-4-12B-it | images |
| GRPO | gemma-4-26B-A4B-it, gemma-4-31B-it | images |

GRPO with images on **Qwen3.6-27B / Qwen3.6-35B-A3B** needed three bug fixes (found on this
branch):

- the RL trainer loaded the text-only processor class instead of the multimodal one;
- prompt-length `mm_token_type_ids` were forwarded together with the prompt+completion `input_ids`;
- Qwen's flattened patch tensor was sliced by row in group expansion and micro-batching.

After the fixes both models run 10 steps with images without error (peak **60.3 GB** per GPU for
Qwen3.6-27B, **82.2 GB** for Qwen3.6-35B-A3B).

**GRPO signal: watch the advantage.** On ScienceQA many groups give all-identical rewards — the
model gets all 4 samples right, or none finishes within `max_new_tokens`. Identical rewards give
zero advantage, so that step is skipped, and **a run where every step is skipped is refused
(exit 96, REFUSED — "vacuous run")**. Measured: gemma-4-12B-it completed 3 of 20 steps with group
size 4 / 4 prompts per step. Remedies: larger `--group-size` (8), more `--prompts-per-step`,
harder questions matched to the model, `--max-new-tokens 400+` for reasoning models.

### 300-step SFT + evaluation (gemma-4-12B-it, 16K)

Training: 300 steps, gradient accumulation 4, **unpadded** batches, 16K context, on a train
split made with `prepare_data.py split` (`data/prepared/<src>.train.jsonl`, mixed into
`data/prepared/mixed.jsonl`); the held-out file is `data/prepared/heldout.jsonl`, built with
`cat data/prepared/image.eval.jsonl data/prepared/video.eval.jsonl data/prepared/text.eval.jsonl >
data/prepared/heldout.jsonl`. The held-out set is 171 disjoint rows (110 image, 40 video,
21 text). Training loss **1.11 → 0.43**, exit 0.

Held-out loss (`eval_heldout_loss.py --data data/prepared/heldout.jsonl`, 0 rows dropped):

| | NLL | perplexity |
|---|---|---|
| base | 1.433 | 4.19 |
| fine-tuned | **0.505** | **1.66** |

`eval_heldout_loss.py` now takes `--video-frames N` — use the same value as
`FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES` in training. Without it, video rows are refused with
exit 96.

Getting the benchmark (the file is public data; the opencompass host's TLS certificate was
expired when tested, so `curl` may fail with "certificate has expired" / error 60):

```bash
curl -L -o data/eval/MMStar.tsv https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv
# if it fails with "certificate has expired", retry with:
curl -k -L -o data/eval/MMStar.tsv https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv
wc -l data/eval/MMStar.tsv    # expect ~1,500 rows; the scorer skips 2 malformed ones
```

Base and fine-tuned are compared on the **same** MMStar rows (both MMStar), one model per GPU
(`CUDA_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=1`) with `--max-tokens 512`.

MMStar (1,498 questions, vLLM, `--max-tokens 512`):

| | MMStar | unparsable |
|---|---|---|
| base | **64.02%** | 112 |
| fine-tuned | 61.88% | 105 |

Specialising on Vietnamese documents and activity videos cost about **2 points** of general
English perception.

### Benchmark scoring lesson (also stated in chapter 05)

An earlier version of `eval_mcq_vllm.py` used `max_tokens=32` and took the **first** standalone
capital letter as the answer. On MMStar that reported **47.00% for the same base model — 17
points too low**: reasoning answers were cut off (26% unparsable), and an English article "A"
could be read as option A. The script now defaults to `--max-tokens 512` and prefers an explicit
answer (`Answer: C`, `the answer is (C)`, `**C**`), then a bare letter, then the last letter on
the final line. **Always check the unparsable count and read a few raw outputs before trusting a
benchmark number**; raise `--max-tokens` to 1024 for reasoning-heavy benchmarks.

## 06 — Ready-to-run configs for each model

`configs/` holds one file per model and context length. Each sets only `MODEL`, `CONTEXT`,
`DATASET` (pointing to `data/prepared/mixed.jsonl`) and `OUTPUT_DIR`; everything else is in
[`scripts/train_sft.sh`](scripts/train_sft.sh).

| config | model | context |
|---|---|---|
| `configs/gemma-4-12B-it_16k.env` / `_32k.env` | google/gemma-4-12B-it | 16384 / 32768 |
| `configs/gemma-4-26B-A4B-it_16k.env` / `_32k.env` | google/gemma-4-26B-A4B-it | 16384 / 32768 |
| `configs/gemma-4-31B-it_16k.env` / `_32k.env` | google/gemma-4-31B-it | 16384 / 32768 |
| `configs/Qwen3.6-27B_16k.env` / `_32k.env` | Qwen/Qwen3.6-27B | 16384 / 32768 |
| `configs/Qwen3.6-35B-A3B_16k.env` / `_32k.env` | Qwen/Qwen3.6-35B-A3B | 16384 / 32768 |

```bash
# SFT on your mixed data
bash examples/vlm-2xH200/scripts/train_sft.sh examples/vlm-2xH200/configs/Qwen3.6-27B_32k.env

# Preference training (text) — scripts/train_dpo.py; see chapter 4
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 \
  examples/vlm-2xH200/scripts/train_dpo.py --model models/gemma-4-12B-it \
  --data data/pref/dpo_train.jsonl --output-dir runs/dpo --algorithm dpo \
  --max-steps 20 --max-length 4096

# RL (images) — scripts/train_grpo.py; see chapter 4
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 \
  examples/vlm-2xH200/scripts/train_grpo.py --model models/gemma-4-12B-it \
  --data data/rl_scienceqa/scienceqa_train.jsonl --output-dir runs/grpo --algorithm grpo \
  --max-steps 20 --max-new-tokens 400 --group-size 4 --prompts-per-step 4
```

Override any setting from the shell with environment variables (`STEPS`, `LR`, `RANK`, `ALPHA`,
`SAVE_EVERY`, `GRAD_ACCUM`, `VIDEO_FRAMES`):
`STEPS=2000 LR=5e-5 bash examples/vlm-2xH200/scripts/train_sft.sh examples/vlm-2xH200/configs/...`.

## Things that will bite you on this VM (all measured)

- `localhost` resolves only to IPv6 `::1` on the VM image, so `torchrun --standalone` hangs.
  Always pass `--master_addr 127.0.0.1`.
- 32K context needs `--fused-loss liger`. Without it, gemma-4-12B-it tries to allocate a 32 GiB
  fp32 logits tensor (32768 x 262144 x 4 bytes) and runs out of memory.
- Qwen3.6 needs `pip install flash-linear-attention` for speed (0.5.2): 502 vs 3,156 tokens/s on
  Qwen3.6-27B at 32K.
- There is no CUDA toolkit (`nvcc`) on the VM. vLLM for evaluation goes in a **separate venv**
  (keeps it from changing the training env):
  `python -m venv vllm-env && vllm-env/bin/pip install vllm` (tested: vLLM 0.30-0.31 with
  torch 2.13); it needs `VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0` and `--tp 1`.

## Sample data

- Image + text: [`trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning`](https://huggingface.co/datasets/trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning)
- Video + text: [`trannhiem/TranNhiem-Action100M-Human-Activities`](https://huggingface.co/datasets/trannhiem/TranNhiem-Action100M-Human-Activities)
  (annotations under the FAIR Noncommercial Research License; videos are third-party YouTube
  content, noncommercial research only). Layout: `data/action100m_split1.jsonl` (plus
  `_shortvideo` / `_longvideo`) and `videos/<id>.mp4`; download with
  `hf download trannhiem/TranNhiem-Action100M-Human-Activities --repo-type dataset --local-dir data/raw/video`
  and convert with `--input data/raw/video/data/action100m_split1.jsonl --video-root data/raw/video/videos`
  (see chapter 02).
  **Publishing in progress: if the dataset is not on the Hub yet, ask the organisers for the files and place them in that layout.**
- Text-only: a 209-row Vietnamese reasoning file supplied by the organisers.