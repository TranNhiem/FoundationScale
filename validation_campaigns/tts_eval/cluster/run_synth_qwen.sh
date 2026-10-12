#!/bin/bash
# T1a: synthesize a Seed-TTS split with Qwen3-TTS Base. usage: run_synth_qwen.sh GPU MODEL_DIRNAME LANG LIST TAG
set -u
G=$1; M=$2; L=$3; LST=$4; T=$5
U=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 1000 ] || { echo "REFUSE gpu $G busy $U"; exit 96; }
E=$HOME/datasets/speech/tts_eval; R=$HOME/Project-Developments/FoundationScale; mkdir -p $E/synth/$T
env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute \
  $HOME/envs/tts-qwen/bin/python $R/validation_campaigns/tts_eval/synth_qwen3tts.py --model $HOME/pretraining_weights/Speech-Models/TTS/$M \
  --testset-dir $E/seedtts_testset --lang $L --list $LST --out $E/synth/$T > $E/synth/$T.log 2>&1
echo "exit=$?" >> $E/synth/$T.log
