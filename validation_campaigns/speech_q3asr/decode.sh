#!/bin/bash
# Decode one manifest with a Qwen3-ASR checkpoint (eval_qwen3_asr.py, base processor); see EVIDENCE.md.
# usage: q3_decode.sh GPU MODEL MANIFEST_NAME OUTDIR   (PROC=<dir> for another size's processor)
# NUM_BEAMS=4 LENGTH_PENALTY=0.8 for beam search (default greedy); BATCH=<n> for the batch size;
# MNT=<n> for max new tokens (one-pass long-form needs more than the default 1024).
CLUSTER_HOME="${CLUSTER_HOME:-$HOME}"  # holds datasets/, pretraining_weights/, envs/, .local/share/enroot
set -u
G=$1; W=$2; MF=$3; O=$4
U=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 1000 ] || { echo "REFUSE gpu $G busy $U"; exit 96; }
R=${FS_TREE:-$CLUSTER_HOME/Project-Developments/FoundationScale}; mkdir -p $O
ENROOT_DATA_PATH=$CLUSTER_HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 -e PYTHONPATH=$R/src --mount $CLUSTER_HOME:$CLUSTER_HOME fs-g4e4b-nemo-automodel-26-04_compute \
  $CLUSTER_HOME/envs/hf-cand-519/bin/python $R/validation_campaigns/speech_repro/eval_qwen3_asr.py --model $W --processor ${PROC:-$CLUSTER_HOME/pretraining_weights/Speech-Models/Qwen3-ASR-1.7B-hf} --manifest $CLUSTER_HOME/datasets/speech/manifests/$MF --out $O/eval.json \
  ${BATCH:+--batch-size $BATCH} ${MNT:+--max-new-tokens $MNT} ${NUM_BEAMS:+--num-beams $NUM_BEAMS} ${LENGTH_PENALTY:+--length-penalty $LENGTH_PENALTY} > $O/log.txt 2>&1
echo "exit=$?" >> $O/log.txt
