# 05: Train a software-engineering agent (SWE)

In this chapter the policy learns to fix bugs in a Python repository with shell tools. Each
episode starts its own container from the task's image: the repository is at `/repo`, the tests
that grade the fix are hidden, and the network is off. The model reads code, runs Python, edits
files with `bash`, and calls `submit`. FoundationScale then restores the hidden tests, runs them,
and scores the episode: 1.0 if every test the fix must repair now passes and no test that passed
before breaks, else 0.0.

By the end you have built a synthetic bug-fix task set, trained the 9B policy on it, and measured
its resolve rate on held-out libraries.

## What we measured

Held-out set: 24 tasks from three libraries the policy never trains on, 4 attempts each (96).
Same engine settings for every row.

| Checkpoint | Resolved |
|---|---|
| Base MiMo-V2.6-Distill-Qwen-9B | 0.604 (58/96) |
| Seed 5, step 12 | 0.750 (72/96) |
| Seed 5, step 20 | 0.792 (76/96) |
| Seed 5, step 30 | **0.813 (78/96)** |
| Seed 7, step 12 | 0.729 (70/96) |

Two seeds agree at step 12 (+13 to +15 points), and seed 5 keeps improving through step 30.
A run with a looser token budget (8,000 tokens per turn, 16,000 per episode) gained less at step
12 (0.688) and trained a third slower, so the config here uses the tighter budget.

## How the tasks are made

`scripts/make_swe_tasks.py` builds tasks the way SWE-smith does, from small libraries that ship
with this example (`swe_libs/`, 15 pure-Python modules with pytest suites):

1. It keeps only the tests that pass on the correct library.
2. It injects one bug at a time by rewriting the syntax tree: a flipped comparison, a swapped
   operator, an off-by-one constant, a dropped `not`.
3. It keeps a bug only if it breaks 1 to 5 tests and leaves at least 3 passing.
4. Each kept bug becomes a task with its own image (`python:3.12-slim` and pytest): the buggy
   library at `/repo` and the tests at `/tests`, which are hidden while the agent works.
5. With `--build`, it starts every image in a sandbox, runs the tests with the bug and again with
   the correct file, and records which tests the fix must repair (`fail_to_pass`) and which must
   keep passing (`pass_to_pass`). Tasks whose tests do not behave are dropped.

`--issue vague` writes the bug report the agent sees as "users report that `f()` sometimes
returns wrong results"; it names the buggy functions but shows no failing test. With the failing
test shown in the report instead, the base model resolved 93% of tasks, which leaves RL little to
learn. Three libraries (`graphs`, `units`, `lru`) are held out for evaluation.

```bash
export PYTHONNOUSERSITE=1
export FS_SANDBOX_DATA_ROOT=/path/to/sandbox/data FS_SANDBOX_IMAGE_STORE=/path/to/sandbox/images \
       FS_SANDBOX_SCRATCH_ROOT=/path/to/sandbox/scratch
python examples/agentic-rl-GB200/scripts/make_swe_tasks.py \
  --libs examples/agentic-rl-GB200/swe_libs --out /path/to/data/swe \
  --per-lib 12 --issue vague --build --jobs 6 --python "$(which python)"
```

Building takes about 15 minutes on one GB200 node (image builds need network access for `pip
install pytest`). It writes `swe_train.jsonl` and `swe_eval.jsonl` (124 and 24 tasks on our
node; counts vary slightly with the machine and Python version, because each candidate bug runs
the tests under a 30-second timeout).

## Train

```bash
MODEL=/path/to/MiMo-V2.6-Distill-Qwen-9B TASKS=/path/to/data/swe/swe_train.jsonl \
OUT=/path/to/run SANDBOX_ROOT=/path/to/sandbox ENGINE_GPU=2 TRAINER_GPU=3 \
  bash examples/agentic-rl-GB200/scripts/train_agentic_swe.sh
```

Settings in `configs/swe_mimo9b_2gpu.json`, all from the measured run:

- `env.backend: "enroot_task"` and `reward.kind: "swe"`: a sandbox per episode, graded by the
  hidden tests.
- `harness.parser: "qwen_xml"`: this model writes Qwen-XML tool calls. With `hermes_json` every
  tool call fails to parse and the base model resolves nothing.
- Up to 20 tool turns, 2,048 tokens per turn and 12,000 per episode; 4 tasks x 8 attempts per
  step, at most 16 sandboxes at once. A step takes about 25 minutes (about 5 rollout, the rest
  training), so 100 steps take about 40 hours.
- `VLLM_USE_FLASHINFER_SAMPLER=0` and `rollout.engine_recovery_retries: 3`: see the note on
  engine hangs below.

If the trainer runs inside a container that has no enroot (as on our GB200 nodes), run the
script on the host with `BROKER=1 TRAINER_PREFIX="enroot start ... <container>"`. The script then
starts a small exec broker on the host, which launches the sandboxes on the trainer's behalf.
It listens only on 127.0.0.1 and requires a token from a file only you can read.

## Evaluate

Serve a checkpoint, then run the held-out set:

```bash
vllm serve /path/to/run/publish/step_000020 --served-model-name policy --port 18441 &
python examples/agentic-rl-GB200/scripts/eval_swe.py --model /path/to/run/publish/step_000020 \
  --base-url http://127.0.0.1:18441 --tasks /path/to/data/swe/swe_eval.jsonl \
  --n 24 --samples 4 --dump transcripts.jsonl
```

It prints the resolve rate as one JSON line; `--dump` keeps every transcript with its test report
so you can read what the agent did. Serve the base model with the same flags to get the
comparison row.

An eval has no engine recovery of its own, so a stall (see below) leaves it waiting. Stop the eval
and the server, start the server again, and re-run the same command with `--resume`: it keeps the
episodes already in the `--dump` file and runs only the rest.

## Engine hangs

On our GB200 nodes vLLM 0.21 sometimes stops producing tokens mid-step: the requests in flight time
out and those episodes come back as infra failures. FoundationScale recovers from it with
`rollout.engine_recovery_retries`: a step whose engine failures exceed `max_infra_rate` restarts
the engine, pushes the current checkpoint and re-runs the step, and a weight push that meets a
hung engine is retried the same way. Use 3 retries rather than 1; a stall can recur on the retry.

What we measured with the engine settings (keep the defaults in `configs/swe_mimo9b_2gpu.json`):

- Turning off the FlashInfer sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`) did not prevent stalls, but
  the run that used it with the default attention backend went 65+ steps without one.
- Do not switch to `--attention-backend FLASH_ATTN`. A single 96-episode eval ran clean with it,
  but every training run that used it lost about 19 of 32 episodes to timeouts within a few
  steps, and the retries timed out on the same count again, so the runs ended.
