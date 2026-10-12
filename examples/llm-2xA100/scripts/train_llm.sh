#!/bin/bash
# Text-only LLM LoRA SFT, 2x A100-80GB (FSDP, bf16, sdpa).
# usage: scripts/train_llm.sh CONFIG_FILE
#   CONFIG_FILE sets MODEL, CONTEXT, DATASET, OUTPUT_DIR (see configs/*.env)
set -euo pipefail
source "$1"
: "${MODEL:?}" "${CONTEXT:?}" "${DATASET:?}" "${OUTPUT_DIR:?}"
STEPS=${STEPS:-1000}; LR=${LR:-1e-4}; RANK=${RANK:-16}; ALPHA=${ALPHA:-32}
SAVE_EVERY=${SAVE_EVERY:-200}; GRAD_ACCUM=${GRAD_ACCUM:-8}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Declare what the data contains: FoundationScale never guesses a column.
export FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN=conversations
export FOUNDATIONSCALE_TRAIN_OVERLONG=drop                # rows longer than CONTEXT are dropped, never truncated

# Families FS has no FamilySpec for (qwen2, llama, ...) must name LoRA targets explicitly.
TARGET_ARGS=(); for t in ${ADAPTER_TARGETS:-}; do TARGET_ARGS+=(--adapter-target "$t"); done

# Always pass 127.0.0.1: some images resolve localhost to ::1 first and the rendezvous hangs.
# --fused-loss is opt-in (FUSED_LOSS=liger): FS refuses it for families liger is not registered for.
torchrun --nnodes 1 --nproc_per_node 2 --master_addr 127.0.0.1 --master_port ${MASTER_PORT:-29882} \
  -m foundationscale.train \
  --model "$MODEL" --dataset "$DATASET" --output-dir "$OUTPUT_DIR" \
  --nodes 1 --gpus-per-node 2 --dp 2 --profile-name local-single-node \
  --max-steps "$STEPS" --per-device-batch-size 1 --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$LR" --warmup-steps 20 --lr-scheduler-type cosine \
  --max-sequence-length "$CONTEXT" --precision bf16 \
  --adapter lora --adapter-rank "$RANK" --adapter-alpha "$ALPHA" \
  --sharding-strategy fsdp --gradient-checkpointing true --attn-implementation sdpa \
  ${FUSED_LOSS:+--fused-loss "$FUSED_LOSS"} --logging-steps 10 --save-interval "$SAVE_EVERY" "${TARGET_ARGS[@]}"
