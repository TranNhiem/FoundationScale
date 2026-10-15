#!/bin/bash
# LoRA SFT on 2 GPUs of a shared GB200 tray: GB200/shared-machine settings, then the
# unchanged vlm-2xH200 launcher (examples/vlm-2xH200/scripts/train_sft.sh).
# usage: examples/vlm-2xGB200/scripts/train_sft_gb200.sh CONFIG_FILE
set -euo pipefail
: "${1:?usage: train_sft_gb200.sh CONFIG_FILE}"
# Shared accounts: packages in ~/.local leak into every env (README, Step 1).
export PYTHONNOUSERSITE=1
# Keep torchrun's scratch files in your own directory instead of /tmp.
mkdir -p tmp && export TMPDIR=${TMPDIR_OVERRIDE:-$PWD/tmp}
# Two GPUs with no process on them, unless you chose them yourself.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 1000 {print $1}' | head -2 | paste -sd,)
  if [ "$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' ' | wc -w)" -lt 2 ]; then
    echo "fewer than 2 idle GPUs on this tray (found: '${CUDA_VISIBLE_DEVICES}'); try again later" >&2
    exit 96
  fi
  export CUDA_VISIBLE_DEVICES
fi
# The default 29500 is often taken on a shared tray.
export MASTER_PORT=${MASTER_PORT:-29871}
echo "GB200: GPUs $CUDA_VISIBLE_DEVICES, master port $MASTER_PORT, TMPDIR $TMPDIR"
exec bash "$(dirname "$0")/../../vlm-2xH200/scripts/train_sft.sh" "$1"
