# VLM LoRA on 2x H200: evidence

Machine: one VM, 2x NVIDIA H200 (141 GB each), driver 580 / CUDA 13.0, 44 cores, 472 GB RAM.
Stack: torch 2.13.0+cu132, transformers 5.18.0, peft 0.21.2, accelerate 1.15.0,
liger-kernel 0.8.4, flash-linear-attention 0.5.2.

## Settings shared by every run

LoRA r16 / alpha 32 (peft default targets via the family registry), `--sharding-strategy fsdp`,
`--precision bf16`, `--gradient-checkpointing true`, `--attn-implementation sdpa`,
`--per-device-batch-size 1`, dp 2, `--fused-loss liger`,
`FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN=conversations`, `FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN=image`,
`FOUNDATIONSCALE_TRAIN_OVERLONG=drop`, `FOUNDATIONSCALE_TRAIN_PAD_TO_MAX_LENGTH=true`
(every batch padded to the full context, so peak memory is the worst case).
Data: 200 rows, 60% image (Vietnamese document pages) / 40% text-only reasoning, 6 steps,
save at step 6. Peak memory is `nvidia-smi` memory.used, the max over both GPUs.

## Image + text matrix (measured 2026-10-09)

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

Tokens/s over 6 steps is noisy (warm-up included); the long-run table replaces it.
Every saved adapter passed the save gate and loads with `PeftModel.from_pretrained`
with no missing keys (checked on gemma-4-12B-it and Qwen3.6-27B).

## Defects these runs exposed (fixed on this branch)

- 32768 tokens OOMed on gemma-4-12B-it allocating exactly 32 GiB = 32768 x 262144 x 4 B
  (fp32 logits). `--fused-loss liger` removes it.
- Under FSDP1 (tied embeddings: all Gemma-4) peft saved a 40-byte adapter.
- peft's FSDP auto-wrap policy read `_no_split_modules`, which names `Gemma4AudioLayer`;
  gemma-4-31B-it and 26B-A4B have no audio tower, so peft raised before step 1.
- Without flash-linear-attention, Qwen3.6-27B at 32768 ran at 502 tokens/s (reference
  PyTorch path for the gated-delta-rule layers); with it, 1461-3156.
- On this VM `localhost` resolves only to `::1`: `torchrun --standalone` hangs. Launch with
  `--master_addr 127.0.0.1`.

## Fused-loss parity (GPU, ~2k tokens, fused vs unfused loss)

relative difference: gemma-4-12B-it 6e-6, gemma-4-26B-A4B-it 4.7e-4, Qwen3.6-27B 0.

## Not yet measured

Video + text (frame budget and segment sampling pending), long runs, the evaluation loop,
and the RL / preference algorithms with LoRA.
