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
WARMUP=${WARMUP:-0}; GROUP_BY_DURATION=${GROUP_BY_DURATION:-0}; SHARDING=${SHARDING:-}
FSDP_STATE_DICT=${FSDP_STATE_DICT:-}

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
  # LoRA keeps gemma-4-E4B-it and qwen2-audio-7b cheap on 2 GPUs; full fine-tunes pass no adapter
  # flags at all. Both families train their audio tower (+ projector) in FULL automatically
  # alongside the LoRA-wrapped language model, with automatic adapter-target scoping -- nothing
  # extra to declare here (see chapter 03).
  EXTRA_ARGS=(--adapter lora --adapter-rank "$RANK" --adapter-alpha "$ALPHA")
fi

# SHARDING (optional, empty by default = plain DDP replication): a full fine-tune of a 7B+ model
# needs its optimizer state sharded across GPUs to fit -- set SHARDING=fsdp (qwen2-audio-7b.env
# does, for its full fine-tune). LoRA and the smaller full fine-tunes fit under plain DDP and leave
# it unset.
if [ -n "$SHARDING" ]; then
  EXTRA_ARGS+=(--sharding-strategy "$SHARDING")
fi

# FSDP_STATE_DICT (optional, only meaningful with SHARDING=fsdp): qwen2-audio-7b.env sets this to
# "full" (also the CLI default -- the config sets it explicitly for documentation). An earlier round
# measured --fsdp-state-dict full failing checkpoint.save_complete (0/876 tensors matched) on
# transformers>=5.18: transformers reverts the load-time module layout on save (checkpoint keys come
# back as audio_tower.*, the live model holds model.audio_tower.*), so FS's save-completeness check
# was comparing against the wrong names. FIXED upstream (origin/main 152f43c): FS now asks
# transformers' own revert_weight_conversion for the on-disk names and narrows saved fp32 master
# weights to bf16 before the dtype check, so --fsdp-state-dict full now saves a normal, directly
# HF-loadable checkpoint with tower_movement correctly measured -- see 03_training.md section 5.
# --fsdp-state-dict sharded also PASSES the save-completeness gates and (as of current origin/main)
# speech.tower_movement reads sharded (DCP) saves too -- both layouts are adjudicated, not just full.
# full stays the recipe's default anyway: sharded still writes raw torch.distributed.checkpoint (DCP)
# shards instead of an HF-loadable safetensors file, which eval_wer.py can't load without a manual
# conversion -- see 03_training.md section 5.
if [ -n "$FSDP_STATE_DICT" ]; then
  EXTRA_ARGS+=(--fsdp-state-dict "$FSDP_STATE_DICT")
fi

# --gradient-checkpointing is deliberately NEVER passed: gemma-4-E4B-it has KV-shared decoder layers that
# refuse it (it breaks use_cache), and this script keeps one uniform code path for every model instead of
# a model-specific exception -- none of them need it at these step counts/batch sizes.
# /etc/hosts lists ::1 before the IPv4 entry for localhost on this VM, so torchrun's rendezvous
# picks the IPv6 address and hangs: always pass 127.0.0.1 explicitly (never "localhost" or "::1").
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
