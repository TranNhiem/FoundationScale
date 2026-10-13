#!/bin/bash
# O0/O1 VoiceBench run. usage: run_vb.sh GPU ASSISTANT SUBSET [LIMIT]
set -u
G=$1; A=$2; S=$3; LIM=${4:-}
U=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 1000 ] || { echo "REFUSE gpu $G busy $U"; exit 96; }
V=$HOME/datasets/speech/voicebench; R=$HOME/Project-Developments/FoundationScale; W=$HOME/pretraining_weights
mkdir -p $V/runs
env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute \
  $HOME/envs/hf-cand-519/bin/python $R/validation_campaigns/omni_eval/voicebench_run.py --assistant $A --subset $S \
  --data-dir $V/data --vb-code $V/code --eval-python $HOME/envs/vb-eval/bin/python \
  --asr-model $W/Speech-Models/Qwen3-ASR-1.7B-hf --llm $W/Alibaba-Qwen/qwen3/Qwen3-4B-Instruct-2507 \
  --out-dir $V/runs ${LIM:+--limit $LIM} < /dev/null > $V/runs/$A-$S.log 2>&1
echo "exit=$?" >> $V/runs/$A-$S.log
