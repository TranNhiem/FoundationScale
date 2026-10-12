#!/bin/bash
# T0: score the Seed-TTS-eval ground truth with the upstream judges. usage: run_gt.sh GPU LANG LIST OUTTAG
set -u
G=$1; L=$2; LST=$3; T=$4
U=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 1000 ] || { echo "REFUSE gpu $G busy $U"; exit 96; }
E=$HOME/datasets/speech/tts_eval; R=$HOME/Project-Developments/FoundationScale; mkdir -p $E/runs
env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=$G -e TORCH_HOME=$HOME/.cache/torch_tts -e HF_HOME=$HOME/.cache/hf_tts_judge -e MODELSCOPE_CACHE=$HOME/.cache/modelscope_tts_judge --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute \
  $HOME/envs/tts-judge/bin/python $R/validation_campaigns/tts_eval/seedtts_score.py --testset-dir $E/seedtts_testset --lang $L --list $LST --gt \
  --seed-code $E/seed-tts-eval-code --wavlm-ckpt $E/gd/wavlm_large_finetune.pth --python $HOME/envs/tts-judge/bin/python \
  --device cuda:0 --out $E/runs/$T.json < /dev/null > $E/runs/$T.log 2>&1
echo "exit=$?" >> $E/runs/$T.log
