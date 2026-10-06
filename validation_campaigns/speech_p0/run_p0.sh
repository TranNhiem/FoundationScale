#!/bin/bash
# P0 speech de-risk probe launcher (docs/research/speech.md section 6, P0).
#
# Attaches to an existing holding job with --overlap (never scancel a holding job to get
# onto its node) and pins ONE GPU inside the container. Every site-specific value is
# configuration; nothing here names a node.
#
#   FS_HOLD_JOBID      required: Slurm job id that holds the tray
#   FS_PROBE_GPU       GPU index inside the allocation (default 0)
#   FS_ENROOT_NAME     unpacked enroot container (default: the thin-plane image)
#   CLUSTER_HOME       root holding weights, datasets and the FS execution folder
#
# Three container traps this launcher encodes, each measured during P0:
#   - srun's default ENROOT_DATA_PATH is node-local scratch, not the unpacked image's home
#   - the $HOME bind mount exposes ~/.local site-packages, which shadow the container's
#     torch/transformers; PYTHONNOUSERSITE=1 turns them off
#   - the container's working directory is read-only, so --out must be absolute
set -euo pipefail

: "${FS_HOLD_JOBID:?FS_HOLD_JOBID is required (the holding job to attach to)}"
CLUSTER_HOME="${CLUSTER_HOME:-$HOME}"
FS_PROBE_GPU="${FS_PROBE_GPU:-0}"
FS_ENROOT_NAME="${FS_ENROOT_NAME:-fs-g4e4b-nemo-automodel-26-04_compute}"
MODEL="${MODEL:-$CLUSTER_HOME/pretraining_weights/Vision-Language-Models/Google/Gemma4/gemma-4-E4B-it}"
LIBRISPEECH="${LIBRISPEECH:-$CLUSTER_HOME/datasets/speech/LibriSpeech}"
OUT="${OUT:-$CLUSTER_HOME/Project-Developments/FoundationScale/artifacts/speech/p0_gemma4_audio.json}"
PROBE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/probe.py"

srun --overlap --jobid="$FS_HOLD_JOBID" --gres=gpu:4 --ntasks=1 \
  env ENROOT_DATA_PATH="$HOME/.local/share/enroot" \
  enroot start \
    -e PYTHONNOUSERSITE=1 \
    -e CUDA_VISIBLE_DEVICES="$FS_PROBE_GPU" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --mount "$CLUSTER_HOME:$CLUSTER_HOME" \
    "$FS_ENROOT_NAME" \
    python3 "$PROBE" --model "$MODEL" --librispeech-root "$LIBRISPEECH" \
      --n 4 --lr 1e-2 --out "$OUT"
