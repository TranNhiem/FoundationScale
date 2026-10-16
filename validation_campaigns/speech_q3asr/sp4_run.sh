#!/bin/bash
# SP-4: Qwen3-ASR-0.6B through the 1.7B recipe (train.sh / decode.sh, same data, steps and selection rule).
# usage: gpu-pool run N --job foxbrain-sp4-q3asr06 -- bash sp4_run.sh   (global batch kept at 16: BS = 16 / N GPUs)
CLUSTER_HOME="${CLUSTER_HOME:-$HOME}"
set -u
GPUS=${CUDA_VISIBLE_DEVICES:?run through gpu-pool}; G0=${GPUS%%,*}
H=$(cd "$(dirname "$0")" && pwd); W=$CLUSTER_HOME/pretraining_weights/Speech-Models/Qwen3-ASR-0.6B-hf; O=$CLUSTER_HOME/datasets/speech/sp4_q3asr06
export PROC=$W; mkdir -p $O
for mf in earnings_validation1000 ood_earnings223000_conv; do
  [ -s $O/base_$mf/eval.json ] || bash $H/decode.sh $G0 $W $mf.jsonl $O/base_$mf
done
N=$(echo $GPUS | tr "," "\n" | wc -l); MODEL=$W BS=$((16 / N)) bash $H/train.sh $GPUS $O/train 750
tail -1 $O/train/log.txt
for ck in $(ls -d $O/train/out/*/ 2>/dev/null | grep -E "[0-9]+/$"); do
  n=$(basename $ck); [ -s $O/val_$n/eval.json ] || bash $H/decode.sh $G0 $ck earnings_validation1000.jsonl $O/val_$n
done
echo "== train+val decodes done $(date)"
