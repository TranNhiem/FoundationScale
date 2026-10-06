#!/bin/bash
# P1 hardware smoke: FoundationScale train() with a declared audio column,
# Gemma-4-E4B full fine-tune, one GPU, five steps. Attaches to a holding job
# with --overlap; every site value is configuration. See ../speech_p0/run_p0.sh
# for the three container traps (ENROOT_DATA_PATH, PYTHONNOUSERSITE, absolute paths).
set -euo pipefail
: "${FS_HOLD_JOBID:?FS_HOLD_JOBID is required (the holding job to attach to)}"
CLUSTER_HOME="${CLUSTER_HOME:-$HOME}"
FS_PROBE_GPU="${FS_PROBE_GPU:-0}"
FS_ENROOT_NAME="${FS_ENROOT_NAME:-fs-g4e4b-nemo-automodel-26-04_compute}"
FS_SRC="${FS_SRC:?FS_SRC is required (src directory of the speech-plane checkout)}"
MODEL="${MODEL:-$CLUSTER_HOME/pretraining_weights/Vision-Language-Models/Google/Gemma4/gemma-4-E4B-it}"
DATASET="${DATASET:?DATASET is required (JSONL rows: text, answer, audio path)}"
OUT="${OUT:?OUT is required (absolute run directory)}"

srun --overlap --jobid="$FS_HOLD_JOBID" --gres=gpu:4 --ntasks=1 \
  env ENROOT_DATA_PATH="$HOME/.local/share/enroot" \
  enroot start \
    -e PYTHONNOUSERSITE=1 -e PYTHONPATH="$FS_SRC" \
    -e CUDA_VISIBLE_DEVICES="$FS_PROBE_GPU" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN=audio -e HF_HUB_OFFLINE=1 \
    --mount "$CLUSTER_HOME:$CLUSTER_HOME" \
    "$FS_ENROOT_NAME" \
    torchrun --standalone --nproc_per_node 1 -m foundationscale.train \
      --model "$MODEL" --dataset "$DATASET" --output-dir "$OUT" \
      --max-steps 5 --per-device-batch-size 2 --learning-rate 1e-5 \
      --max-sequence-length 1024 --precision bf16 --dp 1 --nodes 1 \
      --gpus-per-node 1 --profile-name local-single-node --logging-steps 1 \
      --save-interval 5
