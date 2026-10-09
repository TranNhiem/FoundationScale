# Train your own vision-language model on 2x H200 with FoundationScale

A hands-on path from an empty VM to an evaluated, fine-tuned VLM. It is written for teams that
each get **one machine with 2x NVIDIA H200 (141 GB each)**, and for anyone who wants a worked,
measured example of LoRA training on image + video + text data.

Everything below was run on that machine. The numbers are measured, not estimated; the raw
evidence is in [`validation_campaigns/vlm_2xh200/EVIDENCE.md`](../../validation_campaigns/vlm_2xh200/EVIDENCE.md).

## What you get

| | |
|---|---|
| Models | `Qwen/Qwen3.6-27B` (dense), `Qwen/Qwen3.6-35B-A3B` (MoE), `google/gemma-4-31B-it` (dense), `google/gemma-4-26B-A4B-it` (MoE), `google/gemma-4-12B-it` (dense) |
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

```bash
bash examples/vlm-2xH200/scripts/train_sft.sh examples/vlm-2xH200/configs/gemma-4-12B-it_16k.env
```

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

## 06 — Ready-to-run configs for each model

`configs/` holds one file per model and context length. Each sets only `MODEL`, `CONTEXT`,
`DATASET` and `OUTPUT_DIR`; everything else is in [`scripts/train_sft.sh`](scripts/train_sft.sh).

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

# Preference training (text) and RL (images): see chapter 4
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 \
  examples/vlm-2xH200/scripts/train_dpo.py --model models/Qwen3.6-27B \
  --data data/pref/dpo_train.jsonl --output-dir runs/dpo --algorithm dpo
```

Override any setting from the shell: `STEPS=2000 LR=5e-5 bash scripts/train_sft.sh configs/...`.

## Things that will bite you on this VM (all measured)

- `localhost` resolves only to IPv6 `::1` on the VM image, so `torchrun --standalone` hangs.
  Always pass `--master_addr 127.0.0.1`.
- 32K context needs `--fused-loss liger`. Without it, gemma-4-12B-it tries to allocate a 32 GiB
  fp32 logits tensor (32768 x 262144 x 4 bytes) and runs out of memory.
- Qwen3.6 needs `pip install flash-linear-attention` for speed: 502 vs 3,156 tokens/s on
  Qwen3.6-27B at 32K.
- There is no CUDA toolkit (`nvcc`) on the VM. vLLM evaluation needs
  `VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0` and `--tp 1`.

## Sample data

- Image + text: [`trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning`](https://huggingface.co/datasets/trannhiem/TranNhiem-Vietnamese-DocumentImage-Reasoning)
- Video + text: [`trannhiem/TranNhiem-Action100M-Human-Activities`](https://huggingface.co/datasets/trannhiem/TranNhiem-Action100M-Human-Activities)
  (annotations under the FAIR Noncommercial Research License; videos are third-party YouTube
  content, noncommercial research only)
- Text-only: a 209-row Vietnamese reasoning file supplied by the organisers.
