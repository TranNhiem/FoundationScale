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
| Qwen3.6-27B | 16384 | 0 | 50.9 | 3475 | 512 |
| Qwen3.6-27B | 32768 | 0 | 70.0 | 3078 | 512 |
| Qwen3.6-35B-A3B | 16384 | 0 | 48.3 | 9322 | 320 |
| Qwen3.6-35B-A3B | 32768 | 0 | 57.6 | 8546 | 320 |

The Qwen rows were first measured with the model loaded through `AutoModelForCausalLM`,
which maps `qwen3_5`/`qwen3_5_moe` to text-only classes that skip the vision tower and
ignore pixel inputs, so those runs trained on text alone. The rows above are the re-run
with `AutoModelForImageTextToText` (recorded as `model_auto_class` in each manifest).

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

## Mixed image + video + text matrix (measured 2026-10-09)

300 rows: 120 image, 120 video (Action100M segments, 16 frames per clip sampled uniformly
inside each row's [start, end], passed to the processor as native video), 60 text. Same
settings as above plus `FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN=video`,
`FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES=16`. The pre-pass kept all 300 rows at both contexts.

| model | context | exit | peak GB/GPU | tokens/s (2 GPUs) | LoRA tensors | pre-pass s |
|---|---|---|---|---|---|---|
| gemma-4-12B-it | 16384 | 0 | 32.3 | 2596 | 656 | 196.8 (cold frame cache) |
| gemma-4-12B-it | 32768 | 0 | 49.7 | 1582 | 656 | 32.8 |
| gemma-4-26B-A4B-it | 16384 | 0 | 48.2 | 3626 | 410 | 34.1 |
| gemma-4-26B-A4B-it | 32768 | 0 | 65.7 | 2526 | 410 | 29.0 |
| gemma-4-31B-it | 16384 | 0 | 63.0 | 1088 | 820 | 25.9 |
| gemma-4-31B-it | 32768 | 0 | 90.2 | 650 | 820 | 32.5 |
| Qwen3.6-27B | 16384 | 0 | 51.0 | 3372 | 512 | 61.6 |
| Qwen3.6-27B | 32768 | 0 | 70.3 | 3024 | 512 | 63.7 |
| Qwen3.6-35B-A3B | 16384 | 0 | 48.5 | 8009 | 320 | 60.3 |
| Qwen3.6-35B-A3B | 32768 | 0 | 58.0 | 8071 | 320 | 58.1 |

Video tokens per frame at the default processor resolution: gemma-4-12B-it 63, Qwen3.6-27B 40.

## Video reaches the model (control)

Qwen3.6-27B through `train()`, 8 video rows, 2 steps, same initial weights: real frames
0.510 -> 0.403, all-black frames 0.549 -> 0.509. Before the class fix both runs gave the
same losses. Suite: `test_video_reaches_the_model.py` (gemma-4-12B-it and Qwen3.6-27B).

## Longer context

Mixed image + video + text (16 frames per clip), context runs padded to the full length =
worst case.

65536 tokens:

| model | context | exit | peak GB/GPU | tokens/s (2 GPUs) |
|---|---|---|---|---|
| gemma-4-12B-it | 65536 | 0 | 90.3 | 861 |
| gemma-4-26B-A4B-it | 65536 | 0 | 106.6 | 1379 |
| Qwen3.6-27B | 65536 | 0 | 113.8 | 2042 |
| Qwen3.6-35B-A3B | 65536 | 0 | 81.2 | 4830 |
| gemma-4-31B-it | 65536 | — | OUT OF MEMORY | — |

gemma-4-31B-it's maximum on 2x H200 is 32K.

131072 tokens: out of memory on all five models with this setup (would need context
parallelism or CPU offload).

## Algorithms with LoRA

20 steps each, all exit 0:

- DPO with LoRA on all five models.
- IPO, SimPO, CPO, ORPO on Qwen3.6-27B.
- GRPO, Dr-GRPO, DAPO, GSPO, REINFORCE++, RLOO on gemma-4-12B-it with images.
- GRPO with images on gemma-4-26B-A4B-it and gemma-4-31B-it.

GRPO with images on Qwen3.6-27B / 35B-A3B: three bugs found and fixed on the branch
(text-only class loaded by the RL trainer; prompt-length mm_token_type_ids forwarded with
prompt+completion ids; Qwen's flattened patch tensor sliced by row in group expansion and
micro-batching); after the fixes both run 10 steps with images without error (peak
60.3 / 82.2 GB per GPU).

GRPO signal: on ScienceQA many groups give all-identical rewards (the model gets all 4
samples right, or none finishes within max_new_tokens), and identical rewards give zero
advantage, so the step is skipped; a run where every step is skipped is refused (exit 96,
"vacuous run"). Measured: gemma-4-12B-it measured 3 of 20 steps with group 4 / 4 prompts.
Remedies: larger `--group-size` (8), more `--prompts-per-step`, harder questions matched to
the model, `--max-new-tokens 400+` for reasoning models.

## Longer SFT run and evaluation

gemma-4-12B-it, 16K, 300 steps, grad accumulation 4, unpadded, on a train split made with
`prepare_data.py split`; held-out = 171 disjoint rows: 110 image, 40 video, 21 text.
Training loss 1.11 -> 0.43, exit 0.

Held-out NLL: base 1.433 (perplexity 4.19) -> fine-tuned 0.505 (perplexity 1.66),
0 rows dropped.

MMStar (1,498 questions, vLLM, `--max-tokens 512`): base 64.02% (112 unparsable) ->
fine-tuned 61.88% (105 unparsable): specialising on Vietnamese documents and activity
videos cost about 2 points of general English perception.

`eval_heldout_loss.py` now takes `--video-frames N` (use the same value as
`FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES` in training); without it, video rows are refused with
exit 96.

## Scoring defect found

Must be stated in chapter 05: an earlier version of `eval_mcq_vllm.py` used `max_tokens=32`
and took the FIRST standalone capital letter as the answer. On MMStar that reported 47.00%
for the same base model, 17 points too low: reasoning answers were cut off (26% unparsable),
and an English article "A" could be read as option A. The script now defaults to
`--max-tokens 512` and prefers an explicit answer ("Answer: C", "the answer is (C)",
"**C**"), then a bare letter, then the last letter on the final line. Always check the
unparsable count and read a few raw outputs before trusting a benchmark number; raise
`--max-tokens` to 1024 for reasoning-heavy benchmarks.

## Longer training: 300-step LoRA SFT on all five models (measured)

16K context, unpadded, gradient accumulation 4, on a 2,400-row image + video + text train mix
made with `prepare_data.py split` + `mix`. Held-out = 171 rows the models never saw (110 image,
40 video, 21 text). MMStar = all 1,498 questions, vLLM, `--max-tokens 512`.

| model | held-out NLL base | held-out NLL tuned | MMStar base | MMStar tuned |
|---|---|---|---|---|
| gemma-4-12B-it | 1.433 | 0.505 | 64.02% | 61.88% |
| gemma-4-26B-A4B-it | 3.027 | 0.525 | 65.42% | 65.35% |
| gemma-4-31B-it | 1.878 | 0.449 | 72.63% | 69.96% |
| Qwen3.6-27B | 0.822 | 0.366 | 71.50% | 72.63% |
| Qwen3.6-35B-A3B | 0.874 | 0.405 | 69.29% | 68.69% |

Every model fits the target data much better (held-out NLL down 54-83%), while general English
perception (MMStar) moves between -2.7 and +1.1 points. Watch both numbers when you train.

GRPO with images and `--group-size 8 --prompts-per-step 8` (30 steps, ScienceQA train): updates on
13/30 steps for gemma-4-12B-it and 12/30 for Qwen3.6-27B, versus 3/20 with group size 4. Per-step
reward stays noisy over 30 steps (gemma 0.13-0.86, Qwen 0.50-0.83), so treat this as "RL runs and
learns", not as a measured improvement.

## Not yet measured

131072-token context runs beyond this setup (they need context parallelism or CPU offload
to fit).