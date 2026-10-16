# 02: Tasks and rewards

**What you have:** the install from [chapter 01](01_installation.md).

## What one training step does

- **Tasks:** each step draws 4 tasks from the training file and samples 8 episodes per task: 32 trajectories.
- **Episode:** an episode is a tool loop (`native_tool_loop`, `hermes_json` tool-call parser, default tools). It allows at most
  2 model turns and 12,000 response tokens. The model writes ABC notation and submits it.
- **Reward:** the reward converts the ABC with `abc2midi` and scores the piece in [0, 1].
  - Text with no ABC scores 0, noted `no_abc` in the log.
  - ABC that fails to convert scores 0, noted `rejected`.
  - A sample abstains (no reward, never counted as 0) only if the scorer itself fails.
- **Advantage:** within each task's 8 samples, the advantage is the reward minus the group mean.
  A task whose 8 samples all score the same teaches nothing that step; the log counts the tasks with a spread as `groups_with_spread`.

## Step 1: generate task files

A task is one JSON line: a `uid` and the chat `messages` that open the episode.

```bash
python examples/agentic-rl-GB200/scripts/make_music_tasks.py --out-dir /path/to/data
```

Expected output:

```
/path/to/data/music_train.jsonl: 900 tasks
/path/to/data/music_eval.jsonl: 100 tasks
```

Each prompt asks for one piece with an instrumentation, key, tempo, metre, length and texture. The prompts come from a fixed
vocabulary and a seeded RNG (`--seed 20261008`), so the files are identical on every machine.

**TRAP:** the measured numbers in the [README](README.md) used a different 900/100 split of prompts with the same shape.
Expect different absolute numbers on the generated set; we haven't measured it.

To train on your own prompts, write a JSONL with the same two fields and pass it as `TASKS`. The music reward only needs the
model to submit ABC.

## Step 2: look at the config

[`configs/music_mimo9b_2gpu.json`](configs/music_mimo9b_2gpu.json) is the measured recipe. The launcher fills in its `@...@`
placeholders. The settings that matter:

| Setting | Value | Why |
|---|---|---|
| `policy.algorithm` | `agentic_grpo` | group-centred advantage, token-level ratio, clip 0.2, no KL |
| `trainer.learning_rate` | 1e-6 | one update per step, gradient checkpointing on |
| `sampling.top_p` / `temperature` | 0.95 / 1.0 | the trainer scores under the same distribution (consistent scoring) |
| `engine.weight_sync_mode` | `reload_endpoint` | the engine reloads each new checkpoint in place (about 9 s) |
| `rollout.max_infra_rate` | 0.5 | stops the run if more than half of a step's episodes fail for infrastructure reasons |
| `rollout.engine_recovery_retries` | 1 | restarts a hung engine and re-runs the step ([chapter 03](03_training.md)) |
| `stability` | best_k 3, drop_from_peak 0.5 | keeps the 3 best checkpoints and flags a collapse |

FS refuses some configs instead of training badly:

- `trainer.max_steps > 1` without `rollout.max_infra_rate`. Use about 0.25 for single-turn tasks and 0.3-0.5 for sandboxed multi-turn tasks.
- An `engine.command` that serves a different model than `policy.model_path`.
- `min_p > 0`, or presence or repetition penalties. The trainer cannot score these consistently.

## Check it worked

```bash
head -c 200 /path/to/data/music_train.jsonl
```

Expected output: a line starting `{"uid": "tr-0000", "messages": [{"role": "system", ...`.

Next: [03_training.md](03_training.md).
