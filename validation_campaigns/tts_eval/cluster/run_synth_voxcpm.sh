#!/bin/bash
# T1b: synthesize a Seed-TTS split with VoxCPM2. usage: run_synth_voxcpm.sh GPU LANG LIST TAG
set -u
G=$1; L=$2; LST=$3; T=$4
U=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 1000 ] || { echo "REFUSE gpu $G busy $U"; exit 96; }
E=$HOME/datasets/speech/tts_eval; R=$HOME/Project-Developments/FoundationScale; mkdir -p $E/synth/$T
env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute \
  $HOME/envs/tts-voxcpm/bin/python $R/validation_campaigns/tts_eval/synth_voxcpm2.py --model $HOME/pretraining_weights/Speech-Models/TTS/VoxCPM2 \
  --testset-dir $E/seedtts_testset --lang $L --list $LST --out $E/synth/$T ${LORA:+--lora-ckpt $LORA} < /dev/null > $E/synth/$T.log 2>&1
echo "exit=$?" >> $E/synth/$T.log
