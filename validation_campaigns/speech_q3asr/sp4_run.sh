#!/bin/bash
# SP-4: Qwen3-ASR-0.6B through the 1.7B recipe (train.sh / decode.sh, same data, steps and selection rule).
# usage: gpu-pool run 2 --job foxbrain-sp4-q3asr06 -- bash sp4_run.sh   (GPU 1 trains with GPU 2; base decodes run first)
CLUSTER_HOME="${CLUSTER_HOME:-$HOME}"
set -u
GPUS=${CUDA_VISIBLE_DEVICES:?run through gpu-pool}; G0=${GPUS%%,*}
H=$(cd "$(dirname "$0")" && pwd); W=$CLUSTER_HOME/pretraining_weights/Speech-Models/Qwen3-ASR-0.6B-hf; O=$CLUSTER_HOME/datasets/speech/sp4_q3asr06
export PROC=$W; mkdir -p $O
for mf in earnings_validation1000 ood_earnings223000_conv; do
  [ -s $O/base_$mf/eval.json ] || bash $H/decode.sh $G0 $W $mf.jsonl $O/base_$mf
done
MODEL=$W bash $H/train.sh $GPUS $O/train 750
tail -1 $O/train/log.txt
for ck in $(ls -d $O/train/out/*/ 2>/dev/null | grep -E "[0-9]+/$"); do
  n=$(basename $ck); [ -s $O/val_$n/eval.json ] || bash $H/decode.sh $G0 $ck earnings_validation1000.jsonl $O/val_$n
done
echo "== train+val decodes done $(date)"
