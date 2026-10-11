#!/bin/bash
# Level 3 of the upstream upgrade process: reproduce every `supported` registry model against its
# model-card reference, each in its own backend container. Exit 0 all PASS, 5 any REGRESSION.
# Usage: run_levels.sh SRC_DIR OUT_DIR   (SRC_DIR: the FoundationScale src/ to test; one free GPU per step)
set -u
SRC=$1; OUT=$2; mkdir -p "$OUT"
FS=/home/hhri-ai/hh28144/Project-Developments/FoundationScale
M=/home/hhri-ai/hh28144/datasets/speech/manifests
WEIGHTS=/home/hhri-ai/hh28144/pretraining_weights/Speech-Models
NORM=$WEIGHTS/whisper-large-v3/normalizer.json
ENR="env ENROOT_DATA_PATH=/home/hhri-ai/hh28144/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e HF_HOME=/home/hhri-ai/hh28144/.cache/huggingface --mount /home/hhri-ai/hh28144:/home/hhri-ai/hh28144 -e PYTHONPATH=$SRC"
NEMO_PY="fs-nemo-26-08 /home/hhri-ai/hh28144/envs/nemo-speech-asr/bin/python"
# HF steps run in the python of their upstream profile (levels.DecodeStep.profile).
declare -A HF_PY=(
  [hf-26.04]="fs-g4e4b-nemo-automodel-26-04_compute python3"
  [hf-cand-519]="fs-g4e4b-nemo-automodel-26-04_compute /home/hhri-ai/hh28144/envs/hf-cand-519/bin/python"
)
free_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000{print $1; exit}'; }

# The container prints a banner on stdout, so the plan is written by Python, not redirected.
$ENR fs-nemo-26-08 /home/hhri-ai/hh28144/envs/nemo-speech-asr/bin/python -c "
import json; from foundationscale.upstream.levels import level3_plan
json.dump([s.__dict__ for s in level3_plan()], open('$OUT/plan.json', 'w'))" > "$OUT/plan.log" 2>&1 \
  || { echo "REFUSE: could not build the plan (see $OUT/plan.log)"; exit 96; }
N=$(python3 -c "import json;print(len(json.load(open('$OUT/plan.json'))))")
echo "plan: $N step(s)"
for i in $(seq 0 $((N-1))); do
  eval "$(python3 - "$OUT/plan.json" "$i" "$M" "$WEIGHTS" "$OUT" <<'PY'
import json, shlex, sys
plan, i, M, W, OUT = json.load(open(sys.argv[1])), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
s = plan[i]
# NeMo steps load the registry upstream ref; HF steps run offline from local weights of that name.
model = f"{W}/{s['upstream_ref'].split('/')[-1]}" if s["backend"] == "hf" else s["upstream_ref"]
out = f"{OUT}/{s['model_id']}.json"
args = [a.format(model=model, manifest=f"{M}/{s['manifest']}", out=out) for a in s["args"]]
print(f"MID={shlex.quote(s['model_id'])} BACKEND={s['backend']} PROFILE={shlex.quote(s['profile'])} EP={shlex.quote(s['entry_point'])} OUTJ={shlex.quote(out)} CARD={s['card_value']} ARGS={shlex.quote(' '.join(args))}")
PY
)"
  G=$(free_gpu); [ -n "$G" ] || { echo "REFUSE: no idle GPU for $MID"; exit 96; }
  if [ "$BACKEND" = nemo ]; then CMD="$ENR -e CUDA_VISIBLE_DEVICES=$G $NEMO_PY -m $EP $ARGS";
  else PY=${HF_PY[$PROFILE]:-}; [ -n "$PY" ] || { echo "REFUSE: no python for profile $PROFILE ($MID)"; exit 96; }
    CMD="$ENR -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 $PY $FS/$EP $ARGS"; fi
  eval "$CMD" > "$OUT/$MID.log" 2>&1 || { echo "[ERROR] $MID decode failed (see $OUT/$MID.log)"; exit 96; }
  $ENR $NEMO_PY $FS/validation_campaigns/speech_repro/score_repro.py "$OUTJ" --whisper-normalizer $NORM --card-wer $CARD > "$OUT/$MID.score.json" 2>/dev/null
done
$ENR $NEMO_PY - "$OUT" <<'PY'
import json, sys
from pathlib import Path
from foundationscale.upstream.levels import judge, level3_plan
out = Path(sys.argv[1]); bad = 0
for step in level3_plan():
    score = json.loads((out / f"{step.model_id}.score.json").read_text().strip().splitlines()[-1])
    v = judge(step, score["wer_whisper_normalizer"]); print(v.line()); bad += not v.passed
sys.exit(5 if bad else 0)
PY
