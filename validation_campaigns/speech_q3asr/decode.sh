#!/bin/bash
# Decode one manifest with a Qwen3-ASR checkpoint (eval_qwen3_asr.py, base processor); see EVIDENCE.md.
# usage: q3_decode.sh GPU MODEL MANIFEST_NAME OUTDIR
set -u
G=$1; W=$2; MF=$3; O=$4
U=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 1000 ] || { echo "REFUSE gpu $G busy $U"; exit 96; }
R=/home/hhri-ai/hh28144/Project-Developments/FoundationScale; mkdir -p $O
ENROOT_DATA_PATH=/home/hhri-ai/hh28144/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 -e PYTHONPATH=$R/src --mount /home/hhri-ai/hh28144:/home/hhri-ai/hh28144 fs-g4e4b-nemo-automodel-26-04_compute \
  /home/hhri-ai/hh28144/envs/hf-cand-519/bin/python $R/validation_campaigns/speech_repro/eval_qwen3_asr.py --model $W --processor /home/hhri-ai/hh28144/pretraining_weights/Speech-Models/Qwen3-ASR-1.7B-hf --manifest /home/hhri-ai/hh28144/datasets/speech/manifests/$MF --out $O/eval.json > $O/log.txt 2>&1
echo "exit=$?" >> $O/log.txt
