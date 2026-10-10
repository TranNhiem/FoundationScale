#!/bin/bash
# LoRA SFT on mixed image + video + text data, 2x H200.
# usage: scripts/train_sft.sh CONFIG_FILE
#   CONFIG_FILE sets MODEL, CONTEXT, DATASET, OUTPUT_DIR (see configs/*.env)
set -euo pipefail
source "$1"
: "${MODEL:?}" "${CONTEXT:?}" "${DATASET:?}" "${OUTPUT_DIR:?}"
STEPS=${STEPS:-1000}; LR=${LR:-1e-4}; RANK=${RANK:-16}; ALPHA=${ALPHA:-32}
SAVE_EVERY=${SAVE_EVERY:-200}; GRAD_ACCUM=${GRAD_ACCUM:-8}; VIDEO_FRAMES=${VIDEO_FRAMES:-16}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Declare what the data contains: FoundationScale never guesses a column.
export FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN=conversations
export FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN=image
export FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN=video
export FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES=$VIDEO_FRAMES   # frames per clip, sampled inside [start, end]
export FOUNDATIONSCALE_TRAIN_OVERLONG=drop                # rows longer than CONTEXT are dropped, never truncated

# localhost is IPv6-only on the competition VM image: always pass 127.0.0.1.
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29500 \
  -m foundationscale.train \
  --model "$MODEL" --dataset "$DATASET" --output-dir "$OUTPUT_DIR" \
  --nodes 1 --gpus-per-node 2 --dp 2 --profile-name local-single-node \
  --max-steps "$STEPS" --per-device-batch-size 1 --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$LR" --warmup-steps 20 --lr-scheduler-type cosine \
  --max-sequence-length "$CONTEXT" --precision bf16 \
  --adapter lora --adapter-rank "$RANK" --adapter-alpha "$ALPHA" \
  --sharding-strategy fsdp --gradient-checkpointing true --attn-implementation sdpa \
  --fused-loss liger --logging-steps 10 --save-interval "$SAVE_EVERY"
