#!/usr/bin/env bash
# Agentic RL on the music task with two GPUs: one serves the policy with vLLM, one trains it.
#
#   MODEL=/path/to/model TASKS=/path/to/music_train.jsonl ABC2MIDI=/path/to/abc2midi \
#   OUT=/path/to/run_dir ENGINE_GPU=0 TRAINER_GPU=1 [PORT=18400] [SEED=6] [STEPS=100] \
#   [ALGORITHM=agentic_grpo] bash train_agentic_music.sh [--dry-run] [--set key=value ...]
# Output goes to the terminal and to $OUT/train.log.
#
# Renders configs/music_mimo9b_2gpu.json with these values into $OUT/config.json, then runs
# `python -m foundationscale.agentic_rl`. Extra arguments go to the CLI unchanged, e.g.
# `--set policy.algorithm=gspo` or `--dry-run` (validate the config and exit without a GPU).
# GPU indices are PHYSICAL indices: the trainer gets CUDA_VISIBLE_DEVICES=$TRAINER_GPU and the
# engine process gets $ENGINE_GPU through the config's engine.env.
set -euo pipefail

: "${MODEL:?set MODEL to the policy checkpoint directory}"
: "${TASKS:?set TASKS to the training task JSONL}"
: "${ABC2MIDI:?set ABC2MIDI to the abc2midi binary}"
: "${OUT:?set OUT to a new run directory}"
: "${ENGINE_GPU:?set ENGINE_GPU (physical index)}"
: "${TRAINER_GPU:?set TRAINER_GPU (physical index)}"
PORT=${PORT:-18400}
SEED=${SEED:-6}
STEPS=${STEPS:-100}
ALGORITHM=${ALGORITHM:-agentic_grpo}

HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
export PYTHONNOUSERSITE=1   # a shared account's ~/.local site-packages must not leak in

python3 - "$HERE/configs/music_mimo9b_2gpu.json" "$OUT/config.json" <<PY
import json, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
for key, value in {
    "@MODEL@": "$MODEL", "@TASKS@": "$TASKS", "@ABC2MIDI@": "$ABC2MIDI", "@OUT@": "$OUT",
    "@PORT@": "$PORT", "@ENGINE_GPU@": "$ENGINE_GPU", "@HOME@": "$HOME",
}.items():
    text = text.replace(key, value)
cfg = json.loads(text.replace('"@PORT_INT@"', "$PORT"))
cfg["trainer"]["seed"] = cfg["rollout"]["seed"] = int("$SEED")
cfg["trainer"]["max_steps"] = int("$STEPS")
cfg["policy"]["algorithm"] = "$ALGORITHM"
json.dump(cfg, open(dst, "w"), indent=2)
print(f"config: {dst}")
PY

# The run log (rollout, stability and recovery lines) is also kept in $OUT/train.log.
CUDA_VISIBLE_DEVICES=$TRAINER_GPU \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -m foundationscale.agentic_rl --config "$OUT/config.json" "$@" 2>&1 | tee -a "$OUT/train.log"
