#!/bin/bash
# STEP: speech_cpt SP-1 -- Qwen3-ASR-0.6B speech continued pre-training on the Earnings-22 pool,
# pseudo-labelled by Qwen3-ASR-1.7B-hf (teacher), then SFT from the CPT model on earnings_train6000.
# Stages: (a) build pool, (b) pseudo-label with decode.sh, (c) cpt.jsonl manifest, (d) CPT from
# 0.6B on cpt.jsonl (1 epoch: rows/16 steps at BS=16 on 1 GPU), (e) cpt_with_processor (checkpoint +
# 0.6B processor symlinks), (f) heldout_loss for base 0.6B and CPT 0.6B on earnings_validation1000
# before SFT (SP-1's safety check -- loss must not skip entirely to see the change), (g) SFT from
# the CPT model on earnings_train6000 (SP-4 recipe: 750 steps, BS=16, save-interval 250),
# (h) decode earnings_validation1000 for each SFT checkpoint with decode.sh (0.6B processor).
# Each stage skipped when its output is present; no rows dropped silently -- every drop is counted
# in the .record.json each python script writes.
# usage: gpu-pool run 1 --job foxbrain-sp1 -- bash run_sp1.sh
set -u
GPUS_CANDIDATE="${CUDA_VISIBLE_DEVICES:?run through gpu-pool}"
G0=${GPUS_CANDIDATE%%,*}
CLUSTER_HOME="${CLUSTER_HOME:-$HOME}"
R="${FS_TREE:-$CLUSTER_HOME/Project-Developments/FoundationScale}"
S="$CLUSTER_HOME/datasets/speech"
W="$CLUSTER_HOME/pretraining_weights/Speech-Models"
O="${1:-$S/sp1_cpt_q3asr06}"
MANIFESTS="$S/manifests"
DATA_DIR="$S/earnings22_robust/data"
POOL=$O/pool.jsonl
TEACHER=${TEACHER:-$W/Qwen3-ASR-1.7B-hf}  # any Qwen3-ASR checkpoint; TEACHER_PROC = its processor
BASE=$W/Qwen3-ASR-0.6B-hf
RUN() {  # enroot + venv runner matching train.sh/decode.sh's container surface
  ENROOT_DATA_PATH="$CLUSTER_HOME/.local/share/enroot" enroot start -e PYTHONNOUSERSITE=1 \
    -e PYTHONPATH="$R/src" -e CUDA_VISIBLE_DEVICES="$GPUS_CANDIDATE" -e HF_HUB_OFFLINE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --mount "$CLUSTER_HOME:$CLUSTER_HOME" fs-g4e4b-nemo-automodel-26-04_compute "$@"
}
mkdir -p "$O"

# (a) build pool
if [ ! -s "$POOL" ]; then
  RUN "$CLUSTER_HOME/envs/hf-cand-519/bin/python" "$R/validation_campaigns/speech_cpt/build_cpt_pool.py" \
    --data-dir "$DATA_DIR" --manifests-dir "$MANIFESTS" --out-dir "$O" || exit $?
fi

# (b) pseudo-label the pool with the teacher decode
if [ ! -s "$O/teacher/eval.json" ]; then
  ln -sf "$POOL" "$MANIFESTS/sp1_pool.jsonl"  # same pool rows for every run  # decode.sh reads a manifest NAME under $MANIFESTS
  PROC="${TEACHER_PROC:-$TEACHER}" bash "$R/validation_campaigns/speech_q3asr/decode.sh" "$G0" "$TEACHER" sp1_pool.jsonl "$O/teacher" || exit $?
fi

# (c) CPT training manifest
if [ ! -s "$O/cpt.jsonl" ]; then
  RUN "$CLUSTER_HOME/envs/hf-cand-519/bin/python" "$R/validation_campaigns/speech_cpt/make_cpt_manifest.py" \
    --pool "$POOL" --pseudo "$O/teacher/eval.json" --out "$O/cpt.jsonl" || exit $?
fi

ROWS=$(wc -l < "$O/cpt.jsonl")
STEPS=$(( (ROWS + 15) / 16 ))   # 1 epoch at BS=16 across one GPU

# (d) CPT from the 0.6B base on cpt.jsonl (the recipe's SFT audio column is already set in train.sh)
if [ ! -d "$O/cpt/out" ]; then
  MODEL=$BASE BS=16 bash "$R/validation_campaigns/speech_q3asr/train.sh" "$GPUS_CANDIDATE" "$O/cpt" "$STEPS"
fi

# (e) symink the last cpt checkpoint + the 0.6B processor as cpt_with_processor
if [ ! -e "$O/cpt_with_processor/config.json" ]; then
  CK="$O/cpt/out/final/"  # the trainer's final save (the step count need not be a multiple of 250)
  [ -d "$CK" ] || { echo "REFUSE no final CPT checkpoint under $O/cpt/out"; exit 96; }
  mkdir -p "$O/cpt_with_processor"
  for f in "$CK"*; do ln -sf "$f" "$O/cpt_with_processor/$(basename "$f")"; done
  # only the processor/tokenizer files come from the base (checkpoints carry weights only)
  for f in processor_config.json tokenizer.json tokenizer_config.json chat_template.jinja; do
    [ -e "$O/cpt_with_processor/$f" ] || ln -s "$BASE/$f" "$O/cpt_with_processor/$f"
  done
fi

# (f) heldout_loss for base 0.6B and CPT 0.6B on the validation manifest (SP-1's safety check)
for tag in base cpt; do
  if [ "$tag" = base ]; then M=$BASE; else M=$O/cpt_with_processor; fi
  [ -s "$O/heldout_${tag}_earnings_validation1000.json" ] || CUDA_VISIBLE_DEVICES=$GPUS_CANDIDATE \
    RUN "$CLUSTER_HOME/envs/hf-cand-519/bin/python" "$R/validation_campaigns/speech_cpt/heldout_loss.py" \
    --model "$M" --processor "$BASE" --manifest "$MANIFESTS/earnings_validation1000.jsonl" \
    --out "$O/heldout_${tag}_earnings_validation1000.json"
done

# (g) SP-4 SFT recipe: 750 steps, BS=16, save-interval 250, from the CPT model
if [ ! -d "$O/sft/out" ]; then
  MODEL=$O/cpt_with_processor BS=16 bash "$R/validation_campaigns/speech_q3asr/train.sh" "$GPUS_CANDIDATE" "$O/sft" 750
fi

# (h) decode validation1000 per SFT checkpoint
for CK in $(ls -d "$O"/sft/out/*/ 2>/dev/null | grep -E '[0-9]+/$'); do
  n=$(basename "$CK")
  [ -s "$O/eval_$n/eval.json" ] || PROC=$BASE bash "$R/validation_campaigns/speech_q3asr/decode.sh" \
    "$G0" "$CK" "earnings_validation1000.jsonl" "$O/eval_$n"
done
echo "== sp1 stages done $(date)"
