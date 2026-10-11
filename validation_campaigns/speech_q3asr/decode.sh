#!/bin/bash
# Decode one manifest with a Qwen3-ASR checkpoint (eval_qwen3_asr.py, base processor); see EVIDENCE.md.
# usage: q3_decode.sh GPU MODEL MANIFEST_NAME OUTDIR
CLUSTER_HOME="${CLUSTER_HOME:-$HOME}"  # holds datasets/, pretraining_weights/, envs/, .local/share/enroot
set -u
G=$1; W=$2; MF=$3; O=$4
U=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 1000 ] || { echo "REFUSE gpu $G busy $U"; exit 96; }
R=$CLUSTER_HOME/Project-Developments/FoundationScale; mkdir -p $O
ENROOT_DATA_PATH=$CLUSTER_HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 -e PYTHONPATH=$R/src --mount $CLUSTER_HOME:$CLUSTER_HOME fs-g4e4b-nemo-automodel-26-04_compute \
  $CLUSTER_HOME/envs/hf-cand-519/bin/python $R/validation_campaigns/speech_repro/eval_qwen3_asr.py --model $W --processor $CLUSTER_HOME/pretraining_weights/Speech-Models/Qwen3-ASR-1.7B-hf --manifest $CLUSTER_HOME/datasets/speech/manifests/$MF --out $O/eval.json > $O/log.txt 2>&1
echo "exit=$?" >> $O/log.txt
