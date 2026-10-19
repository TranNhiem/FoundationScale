# 06: Train a math solver that uses tools

In this chapter the policy solves competition math problems with a Python interpreter: in each
episode it may run code through the `bash` tool (up to 8 tool turns), then gives its final answer
as `\boxed{...}` and calls `submit`. The reward (`reward.kind: "math"`) is 1.0 when the answer is
equivalent to the reference (after normalization and a symbolic check), else 0.0.

By the end you have trained the 9B policy on DeepScaleR problems and measured it on MATH-500 and
AIME 2025, problems it never trains on.

## What we measured

MATH-500: 500 problems x 4 samples. AIME 2025: 30 problems x 8 samples. Mean accuracy, same
engine settings and `qwen_xml` tool parser for every row.

| Checkpoint | MATH-500 | AIME 2025 |
|---|---|---|
| Base MiMo-V2.6-Distill-Qwen-9B | 0.727 | 0.283 |
| Seed 6, step 20 | 0.836 | 0.338 |
| Seed 6, step 60 | 0.849 | 0.379 |
| Seed 6, step 99 (final) | **0.861** | **0.404** |
| Seed 5, step 20 | 0.814 | 0.329 |

Both benchmarks improve throughout the run, and a second seed matches the first at step 20.

**Tools matter most on the hard problems.** An earlier run trained with the wrong tool-call parser
(`hermes_json`), so every tool call failed and the model learned to answer without running code.
Against its own tool-less base (0.693 / 0.221) it reached 0.856 on MATH-500 and 0.354 on AIME 2025
at step 99. On MATH-500 the two runs end close (0.856 vs 0.861). On AIME 2025, where problems
need real computation, working tools add 5 points (0.354 vs 0.404), and the base model gains
6 points just from being able to use them (0.221 vs 0.283).

## Make the tasks

```bash
export PYTHONNOUSERSITE=1
python examples/agentic-rl-GB200/scripts/make_math_tasks.py --out-dir /path/to/data/math
```

It downloads the public datasets (no token needed): DeepScaleR (about 40k training problems, MIT),
MATH-500 and AIME 2025 for evaluation. Training problems that also appear in an eval set are
dropped. Each task carries the reference answer in `metadata.answer`.

## Train

```bash
MODEL=/path/to/MiMo-V2.6-Distill-Qwen-9B TASKS=/path/to/data/math/math_train.jsonl \
OUT=/path/to/run ENGINE_GPU=2 TRAINER_GPU=3 \
  bash examples/agentic-rl-GB200/scripts/train_agentic_math.sh
```

Settings in `configs/math_mimo9b_2gpu.json`, all from the measured run except one:

- `harness.parser: "qwen_xml"`: this model writes Qwen-XML tool calls. With `hermes_json` every
  call fails to parse (135 of 135 in a 50-problem check) and you train a tool-less model.
- 4 problems x 8 attempts per step, up to 8 tool turns and 12,000 tokens per episode, consistent
  scoring (top_p 0.95). A step takes 15 to 45 minutes; our 100-step run took about 31 hours.
- `rollout.engine_recovery_retries: 3`. The measured run used 1; we raised it after SWE runs on the
  same engine showed a stall can recur on the retry (see [chapter 05](05_swe_agent.md)).

## Evaluate

```bash
vllm serve /path/to/run/publish/step_000099 --served-model-name policy --port 18410 &
python examples/agentic-rl-GB200/scripts/eval_math.py --model /path/to/run/publish/step_000099 \
  --base-url http://127.0.0.1:18410 --tasks /path/to/data/math/math500_eval.jsonl --n 500 --samples 4
python examples/agentic-rl-GB200/scripts/eval_math.py --model /path/to/run/publish/step_000099 \
  --base-url http://127.0.0.1:18410 --tasks /path/to/data/math/aime25_eval.jsonl --n 30 --samples 8
```

Serve the base model the same way to get the comparison row. `eval_math.py` defaults to the
`qwen_xml` parser; pass `--parser` if your model writes another format.
