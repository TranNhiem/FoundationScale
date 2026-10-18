#!/usr/bin/env bash
# SWE-agent RL with two GPUs: one serves the policy with vLLM, one trains it; every episode runs
# in its own enroot sandbox built from the task's image (reward.kind=swe, env.backend=enroot_task).
#
#   MODEL=/path/to/model TASKS=/path/to/swe_train.jsonl OUT=/path/to/run_dir \
#   SANDBOX_ROOT=/path/to/sandbox_dir ENGINE_GPU=0 TRAINER_GPU=1 \
#   [PORT=18400] [SEED=5] [STEPS=100] [BROKER=1 TRAINER_PREFIX=...] bash train_agentic_swe.sh [--dry-run]
# Output goes to the terminal and to $OUT/train.log.
#
# Renders configs/swe_mimo9b_2gpu.json into $OUT/config.json and runs
# `python -m foundationscale.agentic_rl`. The sandboxes use $SANDBOX_ROOT/{data,images,scratch}
# (FS_SANDBOX_DATA_ROOT / _IMAGE_STORE / _SCRATCH_ROOT). Run this on the host that has enroot.
#
# BROKER=1 with TRAINER_PREFIX: run this script on the HOST and the trainer inside a container
# without enroot (the GB200 recipe), e.g. TRAINER_PREFIX="enroot start --mount $HOME:$HOME <ctr>".
# The script starts the host exec broker on 127.0.0.1:$((PORT+100)) with a 0600 token file in
# $OUT, exports FS_SANDBOX_BROKER(_TOKEN_FILE), and stops the broker when the run exits. The
# container must see $OUT, $SANDBOX_ROOT, $TASKS and the task images at the same paths, and must
# receive the FS_SANDBOX_* variables (pass them with --env in TRAINER_PREFIX if your runtime does
# not forward the environment).
set -euo pipefail

: "${MODEL:?set MODEL to the policy checkpoint directory}"
: "${TASKS:?set TASKS to the training task JSONL from make_swe_tasks.py}"
: "${OUT:?set OUT to a new run directory}"
: "${SANDBOX_ROOT:?set SANDBOX_ROOT to a directory for sandbox containers and scratch}"
: "${ENGINE_GPU:?set ENGINE_GPU (physical index)}"
: "${TRAINER_GPU:?set TRAINER_GPU (physical index)}"
PORT=${PORT:-18400}
SEED=${SEED:-5}
STEPS=${STEPS:-100}

HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT" "$SANDBOX_ROOT/data" "$SANDBOX_ROOT/images" "$SANDBOX_ROOT/scratch"
export PYTHONNOUSERSITE=1   # a shared account's ~/.local site-packages must not leak in
export FS_SANDBOX_DATA_ROOT=$SANDBOX_ROOT/data
export FS_SANDBOX_IMAGE_STORE=$SANDBOX_ROOT/images
export FS_SANDBOX_SCRATCH_ROOT=$SANDBOX_ROOT/scratch

python3 - "$HERE/configs/swe_mimo9b_2gpu.json" "$OUT/config.json" <<PY
import json, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
for key, value in {
    "@MODEL@": "$MODEL", "@TASKS@": "$TASKS", "@OUT@": "$OUT",
    "@PORT@": "$PORT", "@ENGINE_GPU@": "$ENGINE_GPU", "@HOME@": "$HOME",
}.items():
    text = text.replace(key, value)
cfg = json.loads(text.replace('"@PORT_INT@"', "$PORT"))
cfg["trainer"]["seed"] = cfg["rollout"]["seed"] = int("$SEED")
cfg["trainer"]["max_steps"] = int("$STEPS")
json.dump(cfg, open(dst, "w"), indent=2)
print(f"config: {dst}")
PY

if [ "${BROKER:-0}" = 1 ]; then
  export FS_SANDBOX_BROKER=127.0.0.1:$((PORT + 100))
  export FS_SANDBOX_BROKER_TOKEN_FILE=$OUT/broker.token
  setsid python3 -m foundationscale.agentic_rl.sandbox.broker --port $((PORT + 100)) \
    --token-file "$FS_SANDBOX_BROKER_TOKEN_FILE" > "$OUT/broker.log" 2>&1 < /dev/null &
  BROKER_PID=$!
  trap 'kill -- -$BROKER_PID 2>/dev/null' EXIT
  sleep 3
  kill -0 "$BROKER_PID" || { echo "broker did not start:"; cat "$OUT/broker.log"; exit 4; }
fi

# The run log (rollout, stability and recovery lines) is also kept in $OUT/train.log.
CUDA_VISIBLE_DEVICES=$TRAINER_GPU \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ${TRAINER_PREFIX:-} python3 -m foundationscale.agentic_rl --config "$OUT/config.json" "$@" \
  2>&1 | tee -a "$OUT/train.log"
