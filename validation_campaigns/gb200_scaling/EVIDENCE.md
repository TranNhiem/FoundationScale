# GB200 scaling campaign — measured 2026-09-22

Machine-readable companion: `results.json`. Every number here is read off that file;
nothing in this document is estimated, extrapolated or rounded from a different shape.

Device: `sm_100`, 189471 MiB per GPU, 1200 W cap (enforced == max), 2062 MHz max SM
clock, 4 GPUs per tray. Datasheet BF16 dense peak 2500 TFLOP/s; measured sustained
peak 1503.4 TFLOP/s. MFU is reported against **both**, and the measured peak is the
honest denominator.

Toolchain: torch 2.12.1+cu130, transformers 5.13.0, Python 3.11.

---

## 0. The confound that invalidated the first attempt

The first scaling ladder reported 40% efficiency at 4 GPUs and I spent two hypotheses
on it. Gradient accumulation ×8 — which removes seven of every eight gradient
all-reduces — moved world-2 efficiency only 40.7% → 48.4%, so the communication
hypothesis was refuted. Per-GPU power sat at 443–675 W against a 1200 W cap with SM
clocks pinned at the 2062 MHz maximum, so the power hypothesis was refuted too.

The actual cause: the scheduler's **step cgroup granted 2 CPUs** where the job itself
held 120. Every rank in the ladder was contending for two cores.

Controlled, one variable, same binary and same node, cpuset pinned by hand:

| CPUs | world | step ms | tok/s/GPU | weak-scaling efficiency |
|---|---|---|---|---|
| 128 | 1 | 403.4 | 10153.4 | — |
| 128 | 4 | 390.3 | 10494.2 | 103.4% |
| 2 | 1 | 392.9 | 10426.2 | — |
| 2 | 4 | 1262.8 | 3243.7 | **31.1%** |

**The defect is invisible at world 1** — one rank needs about one CPU to dispatch — so
it only appears once you scale out, where it looks exactly like a fabric problem. This
is why the campaign runs outside the scheduler's step allocation.

---

## 1. Weak scaling, 1 → 8 GPUs

Per-rank batch fixed at 2, so the global batch grows with the world and every rank
does identical per-GPU work at every point. seq 2048, eager, dormant towers frozen.

**gemma-4-E4B, compile off / compile on**

| world | step ms (off / on) | aggregate tok/s (off / on) | efficiency (off / on) |
|---|---|---|---|
| 1 | 403.4 / 305.2 | 10,153 / 13,420 | — |
| 2 | 420.3 / 336.4 | 19,493 / 24,355 | 96.0% / 90.7% |
| 4 | 390.3 / 296.0 | 41,977 / 55,346 | 103.4% / 103.1% |
| 8 | 397.0 / 311.4 | 82,547 / 105,245 | 101.6% / 98.0% |

Efficiency above 100% is run-to-run noise, not superlinearity. The honest reading is
**flat to within about ±4%**, i.e. no measurable scaling loss from 1 to 8 GPUs.

**gemma-4-12B dense** — the eager arm does not fit at this batch at any world size, so
compilation here is not an optimisation but the difference between a run and an OOM.

| world | step ms | aggregate tok/s | TFLOP/s per GPU | MFU vs measured | efficiency |
|---|---|---|---|---|---|
| 1 | 391.2 | 10,469 | 732.2 | 48.70% | — |
| 2 | 417.6 | 19,618 | 686.0 | 45.63% | 93.7% |
| 4 | 412.4 | 39,730 | 694.6 | 46.20% | 94.9% |
| 8 | 434.2 | 75,460 | 659.6 | 43.88% | 90.1% |

The ~10% loss at 8 is the cross-tray hop; it does not appear within a tray.

**Peak measured throughput: 144,297 tok/s aggregate**, 3,821 TFLOP/s, 31.77% of
measured peak — E4B, 8 GPUs, batch 2, gradient accumulation 8, compile on, 94.60 GiB,
728 W.

---

## 2. Gradient accumulation, and a memory effect that is not monotone

Accumulation is the largest single throughput win here: E4B on 4 GPUs goes from
55,346 to 71,348 aggregate tok/s between accumulation 1 and 8.

It is also where compilation stops being free:

| compile | accum 4 | accum 8 |
|---|---|---|
| off | 121.27 GiB | 121.27 GiB |
| on | 87.21 GiB | 94.60 GiB |

With compilation off, activation memory is **flat** across accumulation, which is what
the theory predicts — each micro-batch's graph is freed after its backward. With
compilation on it **grows**, and that growth is what turns a 12B run that fits at
accumulation 1 (156.55 of 189 GiB) into an OOM at accumulation 2.

Diagnostic note, because the obvious suspect is innocent: accumulation alone does not
cost memory. Two runs — flat with compile off, growing with compile on — isolate it.

---

## 3. Sharding: what DDP cannot reach at any width

Under DDP every rank holds a full copy of the parameters, the gradients and the AdamW
state, roughly 8 bytes per parameter. **Adding GPUs does not reduce per-GPU memory**,
so a 26B or 31B checkpoint is unmeasurable on a 189 GiB device no matter how wide the
world is. FSDP2 shards all three.

The trade, measured on E4B at world 4 where both arms run: 84.51 GiB sharded against
106.43 GiB replicated, at 575.3 ms against 390.3 ms. Memory for communication.

**8 GPUs, FSDP2, seq 2048, batch 1** — the first measurements of these checkpoints:

| model | tok/s/GPU | TFLOP/s/GPU | MFU vs measured | peak mem |
|---|---|---|---|---|
| gemma-4-26B-A4B (MoE) | 2,706 | 55.7 | 3.71% | 65.36 GiB |
| gemma-4-31B (dense) | 2,076 | 381.2 | 25.36% | 143.24 GiB |

The MoE returns 30% more tokens per second on 2.2× less memory while running at a
seventh of the dense model's utilisation. That is not "MoE is efficient" — it is a
device left arithmetically idle by a routing- and bandwidth-bound step, and it is the
measured argument for expert parallelism rather than plain data parallelism.

**Gradient checkpointing**, the other memory lever, at batch 2:

| model | no recompute | with recompute |
|---|---|---|
| gemma-4-31B | **OOM** | 42.60 GiB, 15,924 tok/s |
| gemma-4-26B-A4B | 109.03 GiB, 40,816 tok/s | 36.49 GiB, 32,693 tok/s |

Roughly a 3× memory cut for about 20% of throughput, and it converts an OOM into a run.

---

## 4. Context length, and a kernel gap

E4B, 4 GPUs, batch 1, compile off:

| seq | eager | sdpa |
|---|---|---|
| 2048 | 389.3 ms, 71.76 GiB | 409.9 ms, 64.22 GiB |
| 4096 | 414.9 ms, 122.18 GiB | 698.5 ms, 99.20 GiB |
| 8192 | **OOM** | 1912.8 ms, 148.55 GiB |
| 16384 | OOM | OOM |

SDPA is **1.68× slower than eager at 4096** where both fit, and is the only arm that
reaches 8192. So the trade is not "sdpa is the fast path": its value here is purely one
more doubling of context.

The cause is a missing kernel, not a tuning choice. This toolchain has **no
flash-attention, no TransformerEngine, no xformers** installed, and SDPA falls back to
`fmha_cutlassB_bf16_aligned_128x64_k65536_sm80` — an Ampere-generation CUTLASS kernel —
on an `sm_100` device. The cuBLAS GEMMs are Blackwell-native (`nvjet_sm100_*`);
attention is not. An eager OOM at long context should be read as a kernel gap, not a
model limit.

---

## 5. A second model family, and the limit of what that proves

The FSDP2 wrap matches on structure — the children of a module list whose qualified
name ends in `layers` — rather than on class names or attribute paths. Two Qwen3.5
checkpoints sharded through it with **no code change**: 64 blocks and 40 blocks.

They are also five times slower per step at a third of the power:

| model | blocks | step ms | tok/s/GPU | power | FLOP model |
|---|---|---|---|---|---|
| Qwen3.5-27B | 64 | 10,097.9 | 202.8 | 315.1 W | not applicable |
| Qwen3.5-35B-A3B | 40 | 6,690.8 | 306.1 | 300.3 W | not applicable |

`layer_types` on both declares `linear_attention`. These are hybrid linear-attention
models running an unfused recurrence, which is why the device is idle rather than busy.
It also means the `12·L·H·S` term does not describe them, so the harness **refuses to
report TFLOP/s and MFU** for these arms rather than printing a number with two decimal
places that describes nothing. Step time, tokens/s, memory and power are measured
directly and stand.

**Structural reach is not measurement validity.** Reaching a family is necessary and
not sufficient; the harness has to know when its own model stops applying.

---

## 6. Interconnect

Cross-tray NVLink is **unavailable** in this configuration:
`transport/nvls.cc NCCL WARN Cuda failure 801 'operation not supported'`. Multi-node
NVLink needs an IMEX domain spanning both trays, and the two trays were held by
separate scheduler jobs. The fallback transport still reached 98.0% weak-scaling
efficiency at world 8, so at this model and batch size the loss is **not measurable** —
which is a statement about this shape, not a general one.

---

## 7. Input pipeline

295 JSONL files, packed to seq 2048:

| workers | blocked median | batch p90 | tokens/s |
|---|---|---|---|
| 0 | 4.82 ms | 7.79 ms | 895,500 |
| 4 | 0.48 ms | 2.89 ms | 3,531,849 |
| 16 | 0.04 ms | 1.11 ms | 9,146,620 |

Against the fastest GPU consumption in this campaign (144,297 tok/s), the loader at 16
workers supplies **63× the demand**, and blocked time is 0.002% of a step. The tail is
reported alongside the median because a pipeline fails in the tail and a synchronous
all-reduce is starved by the slowest rank — the worst batch observed was 5.76 ms, still
0.3% of a step.

**Control:** an empty corpus exits 96 with a refusal naming the path and the skip
count, rather than reporting the infinite throughput of a loader with nothing to load.
Verified by running it.

**Scope: the text path only.** Image decode is not measured here.

---

## 8. Multimodal steps — the vision tower on the clock

Every number in sections 1–7 is text-only. This section puts images in the batch.
gemma-4-E4B-it, seq 2048 fixed, per-rank batch 1, eager, DDP. Because the sequence
length is fixed, adding images does not add tokens — it *converts* text tokens into
image placeholder tokens, and the table reports both so the two are never conflated.

**world 1**

| images/sample | step ms | tok/s/GPU | image tok | text tok | peak mem |
|---|---|---|---|---|---|
| 0 | 427.7 | 4,788 | 0 | 2048 | 71.76 GiB |
| 1 | 586.3 | 3,493 | 258 | 1790 | 75.97 GiB |
| 2 | 581.3 | 3,523 | 516 | 1532 | 80.24 GiB |
| 4 | 571.3 | 3,585 | 1032 | 1016 | 88.90 GiB |

**world 4**

| images/sample | step ms | tok/s/GPU | aggregate tok/s | peak mem |
|---|---|---|---|---|
| 0 | 518.2 | 3,952 | 15,810 | 86.57 GiB |
| 1 | 674.5 | 3,036 | 12,145 | 90.81 GiB |
| 2 | 678.9 | 3,017 | 12,066 | 95.08 GiB |
| 4 | 664.3 | 3,083 | 12,331 | 103.74 GiB |

Two findings, and the second is the useful one:

1. A multimodal step costs a flat **~37% over text-only** (427.7 → 586.3 ms at world 1,
   518.2 → 674.5 at world 4).
2. **That cost is paid by the first image and almost nothing after it.** One image,
   two images and four images all land within 3% of each other in step time, while
   memory climbs linearly at about **4.3 GiB per image**. So the vision tower is not
   the throughput bottleneck at this scale; a fixed per-step overhead is, and the
   scaling constraint on image count is memory rather than time.

The world-4 text-only arm runs at 83% of world 1, well below the 103% the pure-text
ladder reached, and the reason is declared in the run's own output:
`find_unused_parameters=True` is ON. With no images the vision tower receives no
gradient and the DDP reducer would wait forever for buckets that never arrive, so the
arm pays a full autograd-graph traversal every step. That is the measured price of the
omni-checkpoint workaround, and it is why the framework should derive this flag from
the declaration rather than defaulting it.

**Control:** the loss is reported against `ln(vocab)` on every arm. Leaving image
placeholders unmasked in the labels makes the model try to predict them and drives
loss above `ln(vocab)` — a defect that has shipped in this estate before at loss 19.16
against 12.48. Measured here: 0.0049–0.0127 against 12.4766 on all eight arms, so the
masking is correct and these are steps of real training rather than of nonsense. A
second control sits beside it: a directory with no decodable image REFUSES with exit
96 rather than reporting a multimodal number from a run that had no images; verified
by running it.

Decode note: 34,932 images found, 34,932 decoded, **0 failures** in this subset. That
is not the estate-wide rate — a different corpus measured 21.5% undecodable — so the
decode filter stays load-bearing even though it caught nothing here.

---

## What this campaign does NOT measure

Stated because an unmeasured thing must not be read out of a table that does not
contain it:

- image DECODE cost inside the input pipeline — section 7 measures the text path and
  section 8 excludes decode from the timed step;
- video, and any modality other than still images;
- expert-parallel MoE execution, which section 3 argues for and does not demonstrate;
- any attention kernel with an `sm_100` build, because none is installed.
