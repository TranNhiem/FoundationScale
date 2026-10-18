# Agentic music collapse: which algorithms survive the sampler/trainer mismatch (2026-10-16)

> **Correction (2026-10-18):** these runs used `harness.parser: "hermes_json"`, but this model writes
> Qwen-XML tool calls, so every tool call failed to parse and episodes ran without working tools. The
> reward reads the submitted text, so the comparisons below hold for tool-less episodes. See
> `examples/agentic-rl-GB200/README.md` for the measurement.

Setup for every row: MiMo-V2.6-Distill 9B, music task, lr 1e-6, 4 tasks x 8 samples, top_p 0.95, one update
per step. Eval: 100 held-out prompts x 4 samples, mean reward (the `pass>0` column counts samples above zero,
out of 400). "Raw scoring" scores tokens with raw logits while sampling truncates to the 0.95 nucleus (the
mismatch that `EVIDENCE.md` shows inflating the policy's tail); "consistent scoring" scores them under the
sampled distribution (main b60ce53).

## 60-step benchmark, raw scoring, seed 6 (the seed whose baseline collapses at about step 50)

| algorithm | training | final-checkpoint eval | best-checkpoint eval |
|---|---|---|---|
| agentic_grpo (FS default) | collapsed about step 50 | 0.0 (100-step run) | - |
| dapo | collapsed at step 50 (alarm) | 0.010 (8/400) | 0.339 (step 39) |
| dr_grpo | collapsed after its peak at step 34 | 0.014 (9/400) | - |
| grpo | collapsed after its peak at step 29 | 0.039 (27/400) | - |
| **gspo** | **no collapse** | **0.447 (262/400)** | 0.483 (step 45) |

## GSPO under raw scoring, four seeds, 60 steps

| seed | training | final eval |
|---|---|---|
| 6 | no collapse | 0.447 (262/400) |
| 5 | no collapse | 0.516 (279/400) |
| 4 | no collapse | 0.545 (290/400) |
| 7 | no collapse | 0.557 (304/400) |

Mean 0.516. Every other algorithm collapsed on seed 6. GSPO's sequence-level ratio with its tight clip
bounds how far one update can move any single token's probability, which is the channel the tail inflation
runs through.

## The fix vs the algorithm, at equal length (step 60)

| configuration | seeds | step-60 eval |
|---|---|---|
| agentic_grpo + consistent scoring | 6 / 5 / 4 | 0.599 / 0.592 / 0.619 (mean 0.603) |
| gspo + consistent scoring | 6 / 5 / 7 / 4 | 0.559 / 0.613 / 0.596 / 0.533 (mean 0.575) |
| gspo + raw scoring | 6 / 5 / 4 / 7 | mean 0.516 |
| agentic_grpo + raw scoring | 6 / 5 | collapsed (100-step finals 0.0 / 0.091) |

At 100 steps the consistent-scoring agentic_grpo runs reach 0.653 / 0.651 / 0.628 (seeds 6 / 5 / 4).

## Every algorithm with consistent scoring, seed 6, step 60

The same 60-step benchmark as the raw-scoring table above, with consistent scoring on.

| algorithm | raw scoring (final eval) | consistent scoring (step-60 eval) |
|---|---|---|
| agentic_grpo (FS default) | collapsed (0.0) | 0.599 (seed 6; mean of 3 seeds 0.603) |
| dr_grpo | collapsed (0.014) | 0.588 (337/400) |
| gspo | 0.447 | 0.559 (seed 6; mean of 4 seeds 0.575) |
| dapo | collapsed (0.010) | 0.570 (320/400) |
| rloo | not measured | 0.536 (296/400) |
| grpo | collapsed (0.039) | 0.470 (259/400) |

With consistent scoring no algorithm collapsed: all six finish between 0.470 and 0.603. Every algorithm
that collapsed under raw scoring recovers once scoring matches the sampler, which confirms the mismatch, not
the estimator, as the cause. The default `agentic_grpo` is still the best measured. REINFORCE++ is not in the
table: the plane refuses the estimator-free family (`reinforce_baseline`, `reinforce_pp`, `raft`,
`best_of_n`) whenever a `rollout_source` is set, which every agentic run has. The dry run does not catch that
refusal, so such a run fails at its first step.

## Reading

- The collapse is a sampler/trainer mismatch, and consistent scoring removes it: it is the best configuration
  measured and the remedy to keep.
- Of the algorithms, only GSPO survives the mismatch on its own (4 of 4 seeds), at about 0.09 below the
  consistently scored default at step 60. It is the robust choice when scoring cannot match the sampler
  (for example an engine whose sampling the trainer cannot reproduce).
- Combining GSPO with consistent scoring does not help: over four seeds it averages 0.575 against 0.603 for
  agentic_grpo with consistent scoring (paired seeds 6 / 5 / 4: 0.559 vs 0.599, 0.613 vs 0.592, 0.533 vs
  0.619). Once scoring is consistent the default algorithm is the better choice; GSPO's value is robustness
  when scoring cannot be made consistent.
- Best-checkpoint selection by rollout reward did not beat the final checkpoint once scoring was consistent
  (sc_s5: best step 75 0.618 vs final 0.651; sc_s6: best step 83 0.634 vs final 0.653). It pays off only as
  insurance against a collapse (DAPO: best 0.339 vs final 0.010).

Not measured: algorithms outside FS (CISPO, RLOO, REINFORCE++, PPO with a critic), and multi-turn tasks, where
turn-level credit assignment matters. See the algorithm study plan for that queue.
