#!/bin/bash
# SP-2 stage 1: preference pairs from the SP-4 Qwen3-ASR-0.6B SFT model (checkpoint-250) on earnings_train6000.
# usage: gpu-pool run 1 --job foxbrain-sp2-pairs -- bash run_pairs.sh   (FS_TREE = FoundationScale tree; resumable)
set -u
G=${CUDA_VISIBLE_DEVICES:?run through gpu-pool}; R=${FS_TREE:?}
S=$HOME/datasets/speech; W=$HOME/pretraining_weights/Speech-Models; O=$S/sp2_pref; mkdir -p $O
env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e PYTHONPATH=$R/src -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute \
  $HOME/envs/hf-cand-519/bin/python $R/validation_campaigns/speech_pref/build_pairs.py --model $S/sp4_q3asr06/train/out/checkpoint-250 \
  --processor $W/Qwen3-ASR-0.6B-hf --manifest $S/manifests/earnings_train6000.jsonl --out $O/pairs.jsonl \
  --n-samples ${NS:-8} --resume ${LIMIT:+--limit $LIMIT} < /dev/null > $O/pairs.log 2>&1
echo "exit=$?" >> $O/pairs.log
