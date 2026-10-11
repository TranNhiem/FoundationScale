#!/bin/bash
# Fine-tune Qwen3-ASR in FoundationScale (profile hf-cand-519) on Earnings-22 training calls; see EVIDENCE.md.
# usage: q3_train.sh GPUS(e.g. 2,3) OUTDIR STEPS
set -u
GPUS=$1; O=$2; STEPS=$3; N=$(echo $GPUS | tr ',' '\n' | wc -l)
for g in $(echo $GPUS | tr ',' ' '); do U=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 1000 ] || { echo "REFUSE gpu $g busy $U"; exit 96; }; done
R=/home/hhri-ai/hh28144/Project-Developments/FoundationScale; mkdir -p $O
ENROOT_DATA_PATH=/home/hhri-ai/hh28144/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e PYTHONPATH=$R/src -e CUDA_VISIBLE_DEVICES=$GPUS -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN=audio -e FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE=en -e HF_HUB_OFFLINE=1 --mount /home/hhri-ai/hh28144:/home/hhri-ai/hh28144 fs-g4e4b-nemo-automodel-26-04_compute \
  /home/hhri-ai/hh28144/envs/hf-cand-519/bin/python -m torch.distributed.run --standalone --nproc_per_node $N -m foundationscale.train \
  --model /home/hhri-ai/hh28144/pretraining_weights/Speech-Models/Qwen3-ASR-1.7B-hf \
  --dataset /home/hhri-ai/hh28144/datasets/speech/manifests/earnings_train6000.jsonl \
  --output-dir $O/out --max-steps $STEPS --per-device-batch-size 8 --learning-rate 1e-5 --max-sequence-length 1024 \
  --precision bf16 --dp $N --nodes 1 --gpus-per-node $N --profile-name local-single-node --logging-steps 10 \
  --save-interval 250 --dataloader-num-workers 0 --seed 0 > $O/log.txt 2>&1
echo "exit=$?" >> $O/log.txt
