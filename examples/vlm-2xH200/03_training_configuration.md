# 03 — Training configuration: model, LoRA, context length and memory on 2x H200

This chapter is the configuration layer: one launch command, the flags and data declarations behind it, and the memory story that decides what fits on two H200s. Every number below was measured on the standard VM (2x NVIDIA H200, 141 GB each; torch 2.13.0+cu132, transformers 5.18.0, peft 0.21.2, accelerate 1.15.0, liger-kernel 0.8.4, flash-linear-attention 0.5.2). Anything unproven is marked **TODO(verify)**.

## 1. The launch command

Verified script, `examples/vlm-2xH200/scripts/train_sft.sh` (takes `MODEL_DIR CONTEXT DATASET OUTPUT_DIR [STEPS]`):

```bash
#!/bin/bash
# examples/vlm-2xH200/scripts/train_sft.sh MODEL_DIR CONTEXT DATASET OUTPUT_DIR [STEPS]
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
export FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN=conversations
export FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN=image
export FOUNDATIONSCALE_TRAIN_OVERLONG=drop
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 -m foundationscale.train \
  --model $MODEL_DIR --dataset $DATASET --output-dir $OUTPUT_DIR --nodes 1 --gpus-per-node 2 --dp 2 \
  --profile-name local-single-node --max-steps $STEPS --per-device-batch-size 1 --learning-rate 1e-4 \
  --max-sequence-length $CONTEXT --precision bf16 --adapter lora --adapter-rank 16 --adapter-alpha 32 \
  --sharding-strategy fsdp --gradient-checkpointing true --attn-implementation sdpa --logging-steps 1 \
  --save-interval 500 --fused-loss liger
```

### dotenv / torchrun line

| element | why |
|---|---|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | avoids allocator fragmentation at long context |
| `HF_HUB_OFFLINE=1` | models are local on this VM; no network round trips |
| `torchrun --nnodes 1 --nproc_per_node 2` | one node, two processes = two GPUs |
| `--master_addr 127.0.0.1` | **mandatory on this VM**: `localhost` resolves only to `::1`, so `torchrun --standalone` hangs (see §8) |
| `--master_port 29500` | rendezvous port |

### Model, data and topology

| flag | value | meaning |
|---|---|---|
| `--model` | `$MODEL_DIR` | HF model id or local path |
| `--dataset` | `$DATASET` | HF dataset id, a `.json`/`.jsonl` file, or a directory of them |
| `--output-dir` | `$OUTPUT_DIR` | where checkpoints/adapter land |
| `--nodes 1 --gpus-per-node 2` | | declared world shape |
| `--dp 2` | | data parallelism across both GPUs (no `--tp`/`--pp`/`--ep`/`--cp` here) |
| `--profile-name local-single-node` | | either `--profile-name` or `--profile-path` is required |

### Optimisation settings

| flag | value | meaning |
|---|---|---|
| `--max-steps` | `$STEPS` | training budget |
| `--per-device-batch-size 1` | | batch 1 per GPU (the memory-safe setting; see §6) |
| `--learning-rate 1e-4` | | LoRA learning rate |
| `--max-sequence-length $CONTEXT` | 16384 or 32768 | maximum tokenised sequence length. Caveat from the help text: longer samples are *truncated* to this length by the trainer — the dataset-level `FOUNDATIONSCALE_TRAIN_OVERLONG=drop` guard (§2) is what drops or refuses over-long rows before training, never truncates them. Default is 128 (a historical cap), so never omit this flag by accident |
| `--precision bf16` | | declared precision (`bf16`, `fp16`, `fp32`; `nvfp4` is declarable but `train()` **refuses it (96)** — no backend, never a silent fallback) |
| `--adapter lora --adapter-rank 16 --adapter-alpha 32` | | LoRA instead of full fine-tune (§4) |
| `--sharding-strategy fsdp` | | parameters/gradients/optimizer state sharded across both GPUs — this is what makes a 26B/31B checkpoint reachable at all. `ddp` only replicates (each rank then holds the whole model + AdamW state ≈ 8 bytes/parameter). ZeRO/DeepSpeed are REFUSED (96) |
| `--gradient-checkpointing true` | | activation recomputation. Note: it's a string flag, not `store_true`, so an omitted flag stays distinguishable from explicit `false` |
| `--attn-implementation sdpa` | | supplied at model construction; refused (96) if this transformers loader can't provably accept it |
| `--logging-steps 1` | | one loss point per step — the cadence decides when a loss is first OBSERVED; without it the loop binds `max(1, min(10, max_steps))` and the manifest records the effective value |
| `--save-interval 500` | | checkpoint/save-gate cadence (the 6-step evidence runs used their own cadence and saved at step 6) |
| `--fused-loss liger` | | fused linear cross-entropy — mandatory at 32K (§6) |

Flags present in the CLI but not in this script: `--objective`, `--seed`, `--optimizer`, `--warmup-steps`, `--lr-scheduler-type`, `--gradient-accumulation-steps`, `--adapter-target`, `--adapter-dropout`, `--cpu-optimizer-offload`, `--dry-run`, `--launch-corpus`, `--profile-path`, `--tp/--pp/--ep/--cp`. Several of them (**TODO(verify)** which defaults bind where) apply engine defaults when omitted and record *no claim* in the manifest.

## 2. Declaring the data (environment variables)

The data contract is declared through **required/optional environment variables**, not flags:

| variable | values | note |
|---|---|---|
| `FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN` | column name, e.g. `conversations` | the messages column |
| `FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN` | column name, e.g. `image` | the image column |
| `FOUNDATIONSCALE_TRAIN_OVERLONG` | `drop` \| `refuse` | **required, no default**. What to do with rows whose measured length exceeds `--max-sequence-length` |
| `FOUNDATIONSCALE_TRAIN_PAD_TO_MAX_LENGTH` | `true` \| `false` (optional) | `true` pads every batch to the full context — i.e. the **worst-case** memory/load profile. Use it **once** to prove the worst case fits, then turn it off for real training |

Video frame budget (verified names pending): `FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES` (frame count), `FOUNDATIONSCALE_TRAIN_VIDEO_SAMPLING` (sampling strategy), `FOUNDATIONSCALE_TRAIN_VIDEO_MAX_SIDE` (max image side per frame), `FOUNDATIONSCALE_TRAIN_VIDEO_CACHE_DIR` (**TODO(verify)** these four names — video + text is not yet measured).

### The pre-pass: validate and measure before training

Before any gradient step, the run performs a **pre-pass over every row**: it validates each row and measures its tokenised/loaded length. Throughput measured on document images: **~56 rows/s**.

Rows that come out over-long are handled strictly by `FOUNDATIONSCALE_TRAIN_OVERLONG`:

- `drop` — the row is removed from the run,
- `refuse` — the run stops,

but a row is **never silently truncated** to fit. Truncation would make your reported length a fiction; the declaration is `drop` or `refuse` against a measured length.

## 3. Choosing a model

| model | family | dense / MoE | disk size |
|---|---|---|---|
| Qwen3.6-27B | Qwen3.6 | dense | 55.6 GB |
| Qwen3.6-35B-A3B | Qwen3.6 | MoE (3 B active) | 71.9 GB |
| gemma-4-31B-it | Gemma-4 | dense | 62.5 GB |
| gemma-4-26B-A4B-it | Gemma-4 | MoE (4 B active) | 51.6 GB |
| gemma-4-12B-it | Gemma-4 | dense | 23.9 GB |

**What changes per model: nothing but `--model`.** The same command runs all five — the LoRA target registry, the FSDP wrap and the fused loss adapt per family (see §4 and §6).

One real dependency: for **Qwen3.6** models, install `flash-linear-attention` for speed. Without it, the gated-delta-rule layers fall back to the reference PyTorch path: measured **502 tokens/s** vs 1461–3156 tokens/s on Qwen3.6-27B at 32768 (§8).

## 4. LoRA settings

**Verified configuration: rank 16, alpha 32** (`--adapter-rank 16 --adapter-alpha 32`) with the family-default targets:

- **Targets are chosen automatically per family** when `--adapter-target` is omitted: peft's per-model defaults, adjusted by the FoundationScale family registry so that **vision towers are excluded** and, on the two MoE models, **LoRA goes on attention and shared projections, not on the routed experts**. That saves adapters: **Qwen3.6-35B-A3B saves 320 LoRA tensors**, **gemma-4-26B-A4B-it saves 410** (vs 656 for gemma-4-12B-it, 820 for gemma-4-31B-it, 512 for Qwen3.6-27B) — a big adapter for routed experts buys little and costs save-time and checkpoint RAM/disk.
- **Override**: `--adapter-target PATTERN` is repeatable and collected in order. If the adapter resolves to **zero modules**, the run is **refused after wrapping** — never a silent full fine-tune.
- **Dropout**: `--adapter-dropout DROPOUT` (adequate verified value **TODO(verify)**).
- `--adapter lora` **requires** `--adapter-rank`; omitting `--adapter` entirely gives the explicit default: **full fine-tune**. A missing peft install is a REFUSE (96), never a silent full fine-tune.

## 5. Context length: 16K vs 32K

Image + text matrix, measured 2026-10-09 on 2x H200 with r16/alpha32, FSDP, bf16, gradient checkpointing, sdpa, per-device batch 1, dp 2, `--fused-loss liger`, `FOUNDATIONSCALE_TRAIN_PAD_TO_MAX_LENGTH=true` (every batch padded to full context = worst case), 200 rows (60% Vietnamese document-page images / 40% text-only reasoning), 6 steps, save at step 6. Peak memory is `nvidia-smi` `memory.used`, max over both GPUs.

| model | context | exit | peak GB/GPU | tokens/s (2 GPUs) | LoRA tensors saved |
|---|---|---|---|---|---|
| gemma-4-12B-it | 16384 | 0 | 32.3 | 2658 | 656 |
| gemma-4-12B-it | 32768 | 0 | 48.6 | ~1600 | 656 |
| gemma-4-26B-A4B-it | 16384 | 0 | 48.1 | 3855 | 410 |
| gemma-4-26B-A4B-it | 32768 | 0 | 65.7 | 2528 | 410 |
| gemma-4-31B-it | 16384 | 0 | 62.9 | 1101 | 820 |
| gemma-4-31B-it | 32768 | 0 | 90.2 | 650 | 820 |
| Qwen3.6-27B | 16384 | 0 | 50.8 | 2695 | 512 |
| Qwen3.6-27B | 32768 | 0 | 69.7 | 3156 | 512 |
| Qwen3.6-35B-A3B | 16384 | 0 | 48.8 | 4893 | 320 |
| Qwen3.6-35B-A3B | 32768 | 0 | 64.5 | 2862 | 320 |

**Caveat: tokens/s over 6 steps is noisy** (warm-up included); treat the throughput column as indicative only — the long-run table replaces it. The memory column is the planning number.

Key readings: everything fits at 32K on 141 GB per GPU in this worst-case padding mode (highest peak: 90.2 GB, gemma-4-31B-it). Dense 31B is the tightest and the slowest; the MoE models are the most memory- and throughput-efficient here.

## 6. Memory tuning on 2 GPUs

### `--fused-loss liger` is mandatory at 32K

Without it, **gemma-4-12B-it at 32768 tokens OOMed allocating exactly 32 GiB**: the fp32 logits materialisation is `32768 (seq) × 262144 (vocab) × 4 bytes = 32 GiB`. With `--fused-loss liger` the same run peaks at **48.6 GB/GPU** and passes (exit 0). The fused linear cross-entropy kernel is applied to the model *instance* before peft/FSDP wrap it.

Parity check (GPU, ~2k tokens, fused vs unfused loss, relative difference): gemma-4-12B-it 6e-6, gemma-4-26B-A4B-it 4.7e-4, Qwen3.6-27B 0.

If the installed `liger_kernel` (plus FoundationScale's `gemma4_unified` patch) does not cover the loaded model, the run is **REFUSED (96) naming the `model_type`** — it never trains unfused under a declared fused label. Omitting the flag means no patch and the existing (unfused) forward runs.

### The other levers

| lever | setting | notes |
|---|---|---|
| `--gradient-checkpointing true` | on | activation recomputation: compute for memory |
| `--sharding-strategy fsdp` | `fsdp` | shards parameters, gradients and optimizer state over both GPUs. Under DDP every rank holds the whole model + AdamW state (~8 bytes/parameter), so **adding GPUs never lowers per-GPU memory** |
| batch size + accumulation | `--per-device-batch-size 1` + `--gradient-accumulation-steps N` | the effective batch is `1 × 2 GPUs × N`; raise `N` for bigger effective batches at constant memory |
| `--cpu-optimizer-offload true` | last resort | offloads optimizer state to host RAM. Only valid with `--sharding-strategy fsdp` (REFUSED (96) otherwise — FSDP owns the only offload backend here). Note the switch is wider than its name: it moves parameters and gradients as well as optimizer state |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | on | mitigates allocator fragmentation |

### If you OOM — check in this order

1. Is `--fused-loss liger` set? (Without it, 32K dies on logits alone.)
2. Is `--gradient-checkpointing true` set?
3. Is `--sharding-strategy fsdp` set (and `--dp 2`)?
4. Is `--per-device-batch-size 1`? Raise `--gradient-accumulation-steps` instead of batch size.
5. Is `FOUNDATIONSCALE_TRAIN_PAD_TO_MAX_LENGTH=true` still on? Turn it off after the worst-case proof run.
6. If you are at 32K and only need 16K-compatible data, drop context: 32K costs roughly 30–40 GB more per GPU in the measured table.
7. Add `--cpu-optimizer-offload true` (with FSDP).
8. As a last resort, switch to the smaller model — all measured configs except 32K on 31B leave real headroom.

## 7. Reading the output

**Exit codes:**

| code | meaning |
|---|---|
| `0` | pass |
| `5` | red — **a gate measured a problem** (e.g. an empty adapter, a save gate failure) |
| `95` | unmeasured — the declared check could not be measured this run |
| `96` | refused — **a declaration problem**; the refusal message names it (e.g. `nvfp4` precision without a backend, ZeRO/DeepSpeed sharding, an uncovered `model_type` under `--fused-loss liger`, `--cpu-optimizer-offload true` without FSDP, a LoRA adapter resolving to zero modules, missing peft) |

**Artifacts:**

- `run_manifest.json` in the output dir — the record of declarations, claims and telemetry for the run (including the effective `--logging-steps`).
- **Save-gate `PASS` lines** in the log at each save — the gates that verify the saved adapter is real (non-trivial shape, loads cleanly). A red gate surfaces as exit 5.
- **Adapter checkpoint**: `OUTPUT_DIR/checkpoint-N/adapter_model.safetensors`, where `N` is the step at which it was saved. Every adapter in the evidence matrix passed the save gate.

Loading the adapter:

```python
from peft import PeftModel

adapter = PeftModel.from_pretrained(base_model, "OUTPUT_DIR/checkpoint-N")
```

Verified on gemma-4-12B-it and Qwen3.6-27B: **loads with no missing keys**. (Earlier runs here exposed a fixed defect: under FSDP1 with tied embeddings — all Gemma-4 — peft saved a 40-byte adapter. That is exactly what the save gate catches.)

## 8. Troubleshooting

| symptom | cause / fix |
|---|---|
| `torchrun` hangs at rendezvous | On this VM image `localhost` resolves **only to `::1`** (IPv6-only), so `torchrun --standalone` hangs. Launch with `--master_addr 127.0.0.1` (as the script does) and an explicit `--master_port` |
| `Could not find the transformer layer class to wrap` | A **peft bug**, fixed in FoundationScale: peft's FSDP auto-wrap policy read `_no_split_modules`, which names `Gemma4AudioLayer`; gemma-4-31B-it and gemma-4-26B-A4B have no audio tower, so peft raised before step 1. Re-run on the current branch |
| Qwen3.6 un/surprisingly slow at 32K (e.g. 502 tokens/s on Qwen3.6-27B) | Install **`flash-linear-attention`**. Without it the gated-delta-rule layers run the reference PyTorch path; with it, 1461–3156 tokens/s measured |
| Exit 96 | Read the refusal message — it **names the problem**: the undeclared/impossible option (`nvfp4`, ZeRO/DeepSpeed, offload without FSDP, uncovered `model_type` for liger, zero LoRA modules, missing peft). Fix the declaration; nothing was trained |
| Exit 5 | A **gate measured** something wrong (e.g. 40-byte/empty adapter). Read the gate output in the log and `run_manifest.json` |
| Exit 95 | The check was **unmeasured** this run — it is not a pass; re-run with the measurement path available |
| OOM at 32K without `--fused-loss liger` | The 32 GiB fp32 logits allocation (32768 × 262144 × 4 B). Set `--fused-loss liger` |

Not yet measured on this VM: video + text (frame budget and segment sampling pending), long runs, the evaluation loop, and the RL/preference algorithms with LoRA — treat any of their behaviour as **TODO(verify)**.


**Next:** [04 — Training algorithms](04_training_algorithms.md)
