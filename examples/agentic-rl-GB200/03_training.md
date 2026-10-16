# 03: Training

**What you have:** task files from [chapter 02](02_tasks_and_rewards.md), the model directory, and two free GPUs on one tray.

## Step 1: validate the config without a GPU

```bash
MODEL=/path/to/MiMo-V2.6-Distill-Qwen-9B TASKS=/path/to/data/music_train.jsonl \
ABC2MIDI=/path/to/abc2midi OUT=/path/to/run ENGINE_GPU=2 TRAINER_GPU=3 \
  bash examples/agentic-rl-GB200/scripts/train_agentic_music.sh --dry-run
```

Expected output: `config: /path/to/run/config.json`, then the fully resolved config as JSON, and exit code 0.
A refused config prints the reason and exits 96.

## Step 2: launch

Drop `--dry-run`. Start it under `nohup` (or tmux) so it survives your session:

```bash
MODEL=... TASKS=... ABC2MIDI=... OUT=/path/to/run ENGINE_GPU=2 TRAINER_GPU=3 \
  nohup bash examples/agentic-rl-GB200/scripts/train_agentic_music.sh > /dev/null 2>&1 &
```

The engine (`vllm serve`, `--gpu-memory-utilization 0.5 --max-model-len 16384`) runs on GPU `ENGINE_GPU` and the
trainer on GPU `TRAINER_GPU`.

MEASURED on GB200: about 12 minutes per step, mostly rollout. 100 steps take about 20 h.

Options: `STEPS=60`, `SEED=5`, `ALGORITHM=gspo`, `PORT=18401` (for a second run on the same tray), or any
`--set dotted.key=value`.

## Step 3: read the run log

`/path/to/run/train.log` gets one JSON line per event. Three lines matter:

**`[agentic-rl:rollout]`**, once per step:

```
[agentic-rl:rollout] {"abstained": 0, "groups": 4, "groups_with_spread": 4, "reward_max": 0.847,
  "reward_mean": 0.62, "reward_min": 0.0, "reward_notes": {"rejected": 3}, "rollout_step": 45,
  "terminations": {"stop": 32}, "trajectories": 32, ...}
```

- `reward_mean` rises from about 0.1-0.15 to about 0.6 over 100 steps with this recipe.
- `reward_notes.rejected` counts malformed pieces. A collapse shows up here first, at 25-28 of 32.
  A healthy run stays at roughly 0-10.
- `terminations.length` counts episodes that ran out of tokens. Many in a row also signal trouble.

**`[agentic-rl:stability]`**, once per step: `ema`, `peak`, `best_steps` (the 3 checkpoints kept as the best) and
`alarm`. `alarm: true` means the reward has fallen below half of its peak. The best checkpoints stay on disk.

**`[agentic-rl:engine-recovery]`**: printed when every request in a step failed with an engine error.
MEASURED on GB200: a vLLM engine can hang after an in-place weight reload, with 32 requests running and 0 tokens/s.
With `engine_recovery_retries: 1`, FS stops and restarts the engine, reloads the last published checkpoint and re-runs the step.
We verified this by freezing the engine in the middle of a run, after a checkpoint had been published; the run completed with exit 0.

Exit codes: 0 finished, 5 failed during the run (the reason is printed), 96 refused before training started.

## Traps on a shared tray

**TRAP (GPU indices):** containers such as enroot don't remap GPUs, so `ENGINE_GPU` and `TRAINER_GPU` are *physical* indices.
A launcher that hard-coded 0 put a run on someone else's GPU.

**TRAP (one GPU for both):** to run the engine and the trainer on one GPU, lower the engine's memory share and add `--max-num-seqs 64`.
With the 9B hybrid model at `--gpu-memory-utilization 0.35`-`0.4`, vLLM otherwise refuses with
`max_num_seqs (1024) exceeds available Mamba cache blocks`.

**TRAP (network filesystem):** one tray's network filesystem briefly lost the model directory during a checkpoint write.
FS retries for 10 s; three runs still died with `base_model_dir ... is not a directory`. Move long runs off a tray that does this.

## Check it worked

```bash
ls /path/to/run/publish
grep -c 'agentic-rl:rollout' /path/to/run/train.log
```

Expected output: directories named `step_000010`, `step_000020`, ... (every 10th step is archived), plus the best and latest
steps. The count is the number of steps run so far.

Next: [04_evaluation.md](04_evaluation.md).
