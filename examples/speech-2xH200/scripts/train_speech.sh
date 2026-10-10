#!/bin/bash
# Speech/ASR fine-tuning on LibriSpeech manifests, 2x H200.
# usage: scripts/train_speech.sh CONFIG_FILE
#   CONFIG_FILE sets MODEL, DATASET, OUTPUT_DIR, MODEL_KIND, ADAPTER (see configs/*.env)
set -euo pipefail
source "$1"
# MODEL_KIND is pure shell branching (ctc | seq2seq | audio_llm); ADAPTER may be empty (= full fine-tune).
: "${MODEL:?}" "${DATASET:?}" "${OUTPUT_DIR:?}" "${MODEL_KIND:?}" "${ADAPTER:=}"
STEPS=${STEPS:-500}; LR=${LR:-1e-5}; BATCH=${BATCH:-4}; GRAD_ACCUM=${GRAD_ACCUM:-1}
SAVE_EVERY=${SAVE_EVERY:-500}; RANK=${RANK:-16}; ALPHA=${ALPHA:-32}; GPUS=${GPUS:-2}; WORKERS=${WORKERS:-0}
WARMUP=${WARMUP:-0}; GROUP_BY_DURATION=${GROUP_BY_DURATION:-0}

# Declare what the data contains: FoundationScale never guesses a column.
export FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN=audio          # JSONL column holding the audio file path
export HF_HUB_OFFLINE=1                                  # weights are predownloaded: no network calls at train time
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduces allocator fragmentation on long runs
if [ "$GROUP_BY_DURATION" = "1" ]; then
  # Optional throughput nicety (chapter 03): off by default in the measured recipe below -- it is
  # not what makes any of these models improve or regress, just a padding optimisation.
  export FOUNDATIONSCALE_TRAIN_AUDIO_GROUP_BY_DURATION=1
fi

if [ "$MODEL_KIND" = "seq2seq" ]; then
  # Whisper is seq2seq and refuses to train (exit 96) without a language; this tutorial is English-only.
  export FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE=en
else
  # ctc / audio_llm must not see that variable at all, so make sure it is unset.
  unset FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE
fi

EXTRA_ARGS=()
if [ "$ADAPTER" = "lora" ]; then
  # LoRA keeps gemma-4-E4B-it cheap on 2 GPUs; full fine-tunes pass no adapter flags at all. The
  # gemma4 FamilySpec trains the audio tower + embed_audio projector in FULL automatically
  # alongside the LoRA-wrapped language model -- nothing extra to declare here (see chapter 03).
  EXTRA_ARGS=(--adapter lora --adapter-rank "$RANK" --adapter-alpha "$ALPHA")
fi

# --gradient-checkpointing is deliberately NEVER passed: gemma-4-E4B-it has KV-shared decoder layers that
# refuse it (it breaks use_cache), and this script keeps one uniform code path for every model instead of
# a model-specific exception -- none of them need it at these step counts/batch sizes.
# localhost is IPv6-only on this VM: always pass 127.0.0.1 (never "localhost" or "::1").
# Port 29880 is reserved for this speech tutorial (the VLM tutorial uses 29500 on the same kind of box).
torchrun --nnodes 1 --nproc_per_node "$GPUS" --master_addr 127.0.0.1 --master_port 29880 \
  -m foundationscale.train \
  --model "$MODEL" --dataset "$DATASET" --output-dir "$OUTPUT_DIR" \
  --max-steps "$STEPS" --per-device-batch-size "$BATCH" --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$LR" --warmup-steps "$WARMUP" \
  --max-sequence-length 1024 --precision bf16 \
  --dp "$GPUS" --nodes 1 --gpus-per-node "$GPUS" --profile-name local-single-node \
  --logging-steps 10 --save-interval "$SAVE_EVERY" --dataloader-num-workers "$WORKERS" --seed 0 \
  "${EXTRA_ARGS[@]}"
