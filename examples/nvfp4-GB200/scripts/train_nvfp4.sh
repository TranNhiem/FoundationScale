#!/bin/bash
# Full fine-tune with Megatron-Bridge in bf16 or NVFP4. Run inside the NeMo Framework container.
# usage: scripts/train_nvfp4.sh CONFIG_FILE {bf16|nvfp4}
#
# The two arms run the same recipe, data, seed and batch; the only difference is the
# mixed_precision setting: bf16_mixed, or NVFP4 (fp4=e2m1, fp4_recipe=nvfp4). For NVFP4,
# FP4_PARAM in the config picks where the weights live between steps:
#   FP4_PARAM=false (default)  weights stay bf16, the GEMMs run in FP4. Use this for
#                              fine-tuning: it tracked the bf16 loss in our measurement.
#   FP4_PARAM=true             Megatron-Bridge's stock bf16_with_nvfp4_mixed recipe, which
#                              also stores the weights in FP4. At a fine-tuning learning
#                              rate the updates are smaller than one FP4 step and the model
#                              barely learned (see README).
# Exit codes: 0 trained, 96 refused before training (wrong argument, missing input, no
# Blackwell GPU for nvfp4), otherwise the trainer's own exit code.
set -euo pipefail
refuse() { echo "refused: $*" >&2; exit 96; }
[ $# -eq 2 ] || refuse "usage: train_nvfp4.sh CONFIG_FILE bf16|nvfp4"
CONFIG=$1
ARM=$2

case "$ARM" in
  bf16 | nvfp4) ;;
  *) refuse "precision must be bf16 or nvfp4, got '$ARM'" ;;
esac
[ -f "$CONFIG" ] || refuse "config file '$CONFIG' not found"
# shellcheck disable=SC1090
source "$CONFIG"
FP4_PARAM=${FP4_PARAM:-false}
MIXED_ARGS=("mixed_precision=bf16_mixed")
if [ "$ARM" = nvfp4 ]; then
  MIXED_ARGS=("mixed_precision=bf16_with_nvfp4_mixed")
  case "$FP4_PARAM" in
    true) ;;
    false)
      # The stock recipe with fp4_param/fp4_param_gather off. A recipe name cannot be
      # edited field by field from the command line, so the full block is given; it must
      # come AFTER the name (replacing a string node is allowed, extending the recipe's
      # typed node is not). Every other field matches bf16_with_nvfp4_mixed.
      MIXED_ARGS+=("mixed_precision={_target_:megatron.bridge.training.mixed_precision.MixedPrecisionConfig,bf16:true,params_dtype:torch.bfloat16,autocast_dtype:torch.bfloat16,pipeline_dtype:torch.bfloat16,fp4:e2m1,fp4_recipe:nvfp4,fp4_param:false,fp4_param_gather:false,grad_reduce_in_fp32:true}")
      ;;
    *) refuse "FP4_PARAM must be true or false, got '$FP4_PARAM'" ;;
  esac
fi
RUNNER=${RUNNER:-/opt/Megatron-Bridge/scripts/training/run_recipe.py}
[ -f "$RUNNER" ] || refuse "Megatron-Bridge runner not found at $RUNNER (run inside the NeMo container, or set RUNNER)"
[ -d "$HF_MODEL" ] || refuse "HF_MODEL '$HF_MODEL' is not a directory"
[ -d "$MEGATRON_CKPT" ] || refuse "MEGATRON_CKPT '$MEGATRON_CKPT' missing -- run scripts/convert_hf_to_megatron.py first"
if [ "$ARM" = nvfp4 ]; then
  # NVFP4 GEMMs exist only on Blackwell (compute capability 10.x). Elsewhere TransformerEngine
  # errors out; refuse first and say why.
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  [ "${CC%%.*}" -ge 10 ] 2>/dev/null || refuse "NVFP4 needs a Blackwell GPU (compute capability >= 10.0); this GPU reports '$CC'"
fi

# The runner chdirs into the Megatron-Bridge repo, so relative paths would resolve there.
HF_MODEL=$(realpath "$HF_MODEL")
MEGATRON_CKPT=$(realpath "$MEGATRON_CKPT")
OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR")

# NVFP4's backward weight-gradient GEMM uses the token count as its inner dimension, and
# cuBLAS has no NVFP4 kernel for a count that is only a multiple of 16 (the SFT default
# padding). Both arms pad the same way so the bf16 control sees identical batches.
PAD_TO_MULTIPLE=${PAD_TO_MULTIPLE:-128}

DATA_ARGS=(--dataset "$DATASET")
if [ "$DATASET" = local-jsonl ]; then
  [ -d "${DATA_ROOT:-}" ] || refuse "DATASET=local-jsonl needs DATA_ROOT (a dir with training.jsonl / validation.jsonl)"
  DATA_ARGS+=("dataset.dataset_root=$(realpath "$DATA_ROOT")")
fi

OUT="$OUTPUT_DIR/$ARM"
mkdir -p "$OUT"
export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONNOUSERSITE=1
echo "arm=$ARM fp4_param=$([ "$ARM" = nvfp4 ] && echo "$FP4_PARAM" || echo n/a) recipe=$RECIPE gpus=$NPROC tp=$TP out=$OUT"
set -x
torchrun --nproc_per_node "$NPROC" --master_port "${MASTER_PORT:-29871}" "$RUNNER" \
  --recipe "$RECIPE" "${DATA_ARGS[@]}" \
  -tp "$TP" -sl "$SEQ_LENGTH" -gb "$GLOBAL_BATCH" -mb "$MICRO_BATCH" -ms "$STEPS" --lr "$LR" \
  --pretrained_checkpoint "$MEGATRON_CKPT" --save_dir "$OUT/checkpoints" --save_interval "$STEPS" \
  "tokenizer.tokenizer_model=$HF_MODEL" "${MIXED_ARGS[@]}" "rng.seed=$SEED" \
  "+dataset.dataset_kwargs.pad_seq_length_to_mult=$PAD_TO_MULTIPLE" \
  logger.log_interval=1 "logger.tensorboard_dir=$OUT/tb_logs" ${EXTRA_OVERRIDES:-} 2>&1 | tee "$OUT/train.log"
exit "${PIPESTATUS[0]}"
