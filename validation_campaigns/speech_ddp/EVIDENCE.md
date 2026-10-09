# Speech under data parallelism: throughput and a coverage defect

Hardware: one GB200 tray, GPUs 2-3 (GPUs 0-1 were running another user's job). Date: 2026-10-09.
Gemma-4 E4B, audio full fine-tune, bf16, per-device batch 8, 4 dataloader workers,
`scripts/run_speech_dgx18.sh` (torchrun, `--dp N`).

## 1. Throughput, 150 steps

| GPUs | it/s (steady state) | samples/s | scaling |
|---|---|---|---|
| 1 | about 1.55 | about 12.4 | 1.00x |
| 2 | about 1.35 | about 21.6 | **1.74x (about 87%)** |

Each step is about 13% slower on 2 GPUs: the gradient all-reduce of a full fine-tune. The 4-GPU
point waits for a free tray.

## 2. Defect: the speech gates audited one rank

On 2 GPUs, 150 steps trained 2 x 150 x 8 = 2,400 rows. `speech.audio_row_coverage` and
`speech.audio_placeholder_coverage` still reported **1272/1272**, the 1-GPU count (1,200 trained +
72 prefetched by 4 workers x 2 batches x 8). The final adjudication runs on the writing rank only:
the other ranks abstain before it (#445). So the census was rank 0's counters, and rank 1's rows,
refusals and placeholder checks were never audited. `SharedAudioCoverage` aggregates dataloader
workers within a rank, not ranks.

First fix attempt: an all-reduce inside `finish_speech_run`. It **deadlocked**: rank 0 spun at
100% in NCCL while rank 1 had already exited, because only the writing rank gets there.

Fix: `coverage_after_train` sums the census over the ranks (one float64 `all_reduce(SUM)`;
refusal buckets keyed by `AUDIO_LOAD_REASONS`, so a reason seen on only one rank still adds up).
`train()` calls it right after `Trainer.train()` returns, the last point every rank reaches.
`finish_speech_run` now holds no collective and refuses a missing census.

Verified on the same tray, 2 GPUs, 30 steps: **624/624** audio rows and 624/624 placeholders,
exactly 2 x (30 x 8 + 72). Exit 0, no hang. Tests: injected two-rank sums, a real two-process
gloo all-reduce, and a test that `finish_speech_run` never reaches a collective.
