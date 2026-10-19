# Agentic RL on 2x GB200: music composition with a 9B policy

This example trains a 9B policy with FoundationScale's agentic RL plane on a music-composition task.
In each episode the model uses tools to write a piece in ABC notation and submit it. A reward converts the ABC with
`abc2midi` and scores it in [0, 1]. One GPU serves the policy with vLLM and a second GPU trains it.
After every step the trainer publishes a checkpoint and the engine reloads it in place.

By the end you have a trained policy (100 steps, about 20 h) and its held-out reward.
You also learn to read the run log lines that tell you whether training is healthy.
All numbers were measured on one shared GB200 tray (4 GPUs, 186 GB each, aarch64), using 2 of its GPUs.

**What you have:** two free GPUs on a GB200 tray, the MiMo-V2.6-Distill-Qwen-9B checkpoint on local disk, and
an `abc2midi` binary. Other users take idle GPUs within minutes on a shared tray. Claim your two GPUs and launch right away.

---

## What we measured

Eval: 100 held-out prompts, 4 samples each (400 samples), mean reward. The checkpoint before RL scores **0.137**.

> **Correction (2026-10-18): the numbers below were measured with the wrong tool-call parser.**
> These runs used `harness.parser: "hermes_json"`, but MiMo-V2.6-Distill-Qwen-9B writes Qwen-XML
> tool calls (`<tool_call><function=bash><parameter=command>...`). Every tool call failed to parse,
> so the model trained and was evaluated without working tools. The reward reads the submitted text,
> so the scores and the collapse comparison are still valid, but they measure tool-less episodes.
> This example's config now uses `qwen_xml`; tool-using numbers will replace these once re-measured.
> On 50 MATH-500 problems the base model's tool calls failed 135 of 135 times under `hermes_json`
> and 0 of 66 times under `qwen_xml`. Pick the parser that matches your model's chat template.

**Consistent vs raw scoring** (agentic_grpo, 100 steps):

| Scoring | Seed 6 | Seed 5 | Seed 4 |
|---|---|---|---|
| Consistent (this config) | **0.653** | **0.651** | **0.628** |
| Raw logits (older FS behaviour) | 0.0 (collapsed) | 0.091 (collapsed) | collapsed |

"Consistent scoring" means the trainer scores each token under the distribution vLLM actually sampled from:
top_p 0.95, temperature 1.0. FS turns it on from the config's `sampling` block, so there's nothing to set.
With raw-logit scoring, the policy's low-probability tail inflates step after step until outputs turn malformed.
The rollout reward then falls to about 0 around step 45-55
([evidence](../../validation_campaigns/agentic_rl_collapse/EVIDENCE.md)).

**Algorithms, 60 steps** (choose one with `ALGORITHM=...`):

| Algorithm | Raw scoring | Consistent scoring |
|---|---|---|
| agentic_grpo (default) | collapses | 0.599 / 0.592 / 0.619 (seeds 6 / 5 / 4) |
| gspo | **no collapse on 4 of 4 seeds**: 0.447 / 0.516 / 0.545 / 0.557 | 0.559 / 0.613 / 0.596 / 0.533 (seeds 6 / 5 / 7 / 4; mean 0.575) |
| dapo | collapses (0.010) | 0.570 |
| dr_grpo | collapses (0.014) | 0.588 |
| grpo | collapses (0.039) | 0.470 |
| rloo | not measured | 0.536 |

With consistent scoring no algorithm collapses; agentic_grpo is the best measured. REINFORCE++ and the
other estimator-free losses are refused for agentic runs.
Keep consistent scoring; it is the best configuration measured. gspo is the robust choice when scoring cannot match
the sampler. Full table: [ALGORITHMS.md](../../validation_campaigns/agentic_rl_collapse/ALGORITHMS.md).

---

**Software-engineering agent** ([chapter 05](05_swe_agent.md)): the same plane trains the 9B policy to fix
injected bugs in Python libraries with shell tools, one sandbox per episode, graded by hidden tests. Resolve
rate on 24 held-out tasks x 4 attempts: base 0.604, seed 5 at step 12 / 20 / 30 0.750 / 0.792 / 0.813, seed 7
at step 12 0.729.

**Math with tools** ([chapter 06](06_math_tools.md)): the policy runs Python through the bash tool and submits a
boxed answer. MATH-500 0.727 -> 0.861 and AIME 2025 0.283 -> 0.404 after 100 steps; working tools add most on AIME.

---

## Chapters

| Chapter | You do |
|---|---|
| [01_installation.md](01_installation.md) | Install FoundationScale, vLLM and abcMIDI; check the versions |
| [02_tasks_and_rewards.md](02_tasks_and_rewards.md) | Generate task files; see what an episode and the reward are |
| [03_training.md](03_training.md) | Launch the 2-GPU run; read the run log; recover from a hung engine |
| [04_evaluation.md](04_evaluation.md) | Serve a checkpoint and measure its held-out reward |
| [05_swe_agent.md](05_swe_agent.md) | Build bug-fix tasks, train a software-engineering agent in per-episode sandboxes, measure its resolve rate |
| [06_math_tools.md](06_math_tools.md) | Train a math solver that runs Python through tools; measure MATH-500 and AIME 2025 |

## Quick start

Run these from the repository root. Set the paths and pick two free GPUs first.

```bash
export PYTHONNOUSERSITE=1
python examples/agentic-rl-GB200/scripts/make_music_tasks.py --out-dir /path/to/data
MODEL=/path/to/MiMo-V2.6-Distill-Qwen-9B TASKS=/path/to/data/music_train.jsonl \
ABC2MIDI=/path/to/abc2midi OUT=/path/to/run ENGINE_GPU=2 TRAINER_GPU=3 \
  bash examples/agentic-rl-GB200/scripts/train_agentic_music.sh
```

Then evaluate the newest checkpoint as shown in [chapter 04](04_evaluation.md).

The reference for the plane itself (contracts, gates, the verl adapter) is [AGENTIC_RL.md](../../docs/AGENTIC_RL.md).
The minimal single-file config that the CI dry-run uses lives in [`../agentic_rl/`](../agentic_rl/).
