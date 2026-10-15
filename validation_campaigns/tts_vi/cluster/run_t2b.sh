#!/bin/bash
# T2b training on one GPU. usage: run_t2b.sh GPU
set -u
G=$1; RUN=${2:-t2b_lora}; D=$HOME/datasets/speech/vi_tts
U=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 1000 ] || { echo "REFUSE gpu $G busy $U"; exit 96; }
mkdir -p $D/checkpoints $D/logs
cd $D/voxcpm_code && env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute \
  bash -c "cd $D/voxcpm_code && $HOME/envs/tts-voxcpm/bin/python -m torch.distributed.run --standalone --nproc_per_node 1 scripts/train_voxcpm_finetune.py --config_path $D/configs/$RUN.yaml" < /dev/null > $D/logs/$RUN.log 2>&1
echo "exit=$?" >> $D/logs/$RUN.log
