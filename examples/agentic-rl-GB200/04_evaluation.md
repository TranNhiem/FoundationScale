# 04: Evaluation

**What you have:** a run directory with published checkpoints ([chapter 03](03_training.md)), `music_eval.jsonl`
([chapter 02](02_tasks_and_rewards.md)) and one free GPU.

The eval reuses the training episode loop: the same tools, sampling (top_p 0.95, temperature 1.0) and reward.
So its mean is directly comparable to the rollout reward. It runs 100 held-out prompts x 4 samples.

## Step 1: serve a checkpoint

Pick the newest step directory. The last training step is not always published, so don't assume `step_000100` exists.

```bash
CKPT=$(ls -d /path/to/run/publish/step_* | sort | tail -1)
CUDA_VISIBLE_DEVICES=0 vllm serve $CKPT --served-model-name policy --port 18410 \
  --gpu-memory-utilization 0.85 --max-model-len 16384 &
until curl -sf localhost:18410/health > /dev/null; do sleep 5; done
```

## Step 2: run the eval

```bash
python examples/agentic-rl-GB200/scripts/eval_music.py --model $CKPT --base-url http://127.0.0.1:18410 \
  --tasks /path/to/data/music_eval.jsonl --abc2midi /path/to/abc2midi
```

Expected output: one JSON line. For one of our 100-step runs (seed 5):

```
{"model": ".../publish/step_000100", "tasks": 100, "samples": 4, "measured": 400, "abstained": 0, "mean": 0.65143, "pass_gt0": 360, "max": 0.938}
```

`measured` should be 400; a nonzero `abstained` means the scorer failed on some samples. `pass_gt0` counts samples with any credit.

**TRAP:** evaluate with the current FoundationScale code. An older copy of the reward scorer, from before its size limits were added,
ran CPU-bound for hours on a collapsed model's garbage output.

## Which checkpoint to keep

| Run (consistent scoring) | Best checkpoint by rollout reward | Final checkpoint |
|---|---|---|
| seed 5 | step 75: 0.618 | 0.651 |
| seed 6 | step 83: 0.634 | 0.653 |

With consistent scoring, the final checkpoint was as good as the best-k one or better.
The best-k copies are insurance against a collapse. In a collapsed DAPO run, the best checkpoint scored 0.339 and the final one 0.010.

## Check it worked

Compare against the checkpoint before RL. It should score far below your trained checkpoint (0.137 on our split):

```bash
python examples/agentic-rl-GB200/scripts/eval_music.py --model /path/to/MiMo-V2.6-Distill-Qwen-9B \
  --base-url http://127.0.0.1:18410 --tasks /path/to/data/music_eval.jsonl --abc2midi /path/to/abc2midi
```

Serve the base model on the port first, as in Step 1.

Back to the [README](README.md).
