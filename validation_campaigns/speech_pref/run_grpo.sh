#!/bin/bash
# SP-3: GRPO (dr_grpo) from the SP-4 0.6B SFT model with a WER reward on earnings_train6000, a checkpoint every
# quarter, each decoded on earnings_validation1000; selection and the held-out test as in run_dpo.sh.
# usage: gpu-pool run 1 --job foxbrain-sp3-grpo -- bash run_grpo.sh   (FS_TREE = FoundationScale tree; STEPS=<n>)
set -u
# SFT dir = checkpoint-250 weights + the 0.6B processor files (checkpoints carry weights only)
G=${CUDA_VISIBLE_DEVICES:?run through gpu-pool}; R=${FS_TREE:?}
S=$HOME/datasets/speech; W=$HOME/pretraining_weights/Speech-Models; SFT=$S/sp4_q3asr06/sft250_with_processor
STEPS=${STEPS:-200}; EVERY=$(( STEPS / 4 )); D=$S/sp3_grpo${RUN:+_$RUN}; mkdir -p $D
CT="env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e PYTHONPATH=$R/src -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute"
[ -d $D/final ] || $CT $HOME/envs/hf-cand-519/bin/python $R/validation_campaigns/speech_pref/train_grpo.py --model $SFT --manifest $S/manifests/earnings_train6000.jsonl \
  --save-dir $D --algorithm ${ALGO:-dr_grpo} --max-steps $STEPS --save-every $EVERY --learning-rate ${LR:-1e-6} --group-size ${GS:-8} --prompts-per-step ${PPS:-4} < /dev/null > $D.log 2>&1 || { echo "grpo failed"; tail -8 $D.log; exit 1; }
for ck in $(ls -d $D/step_* $D/final 2>/dev/null); do
  n=$(basename $ck); [ -s $D/val_$n/eval.json ] || PROC=$W/Qwen3-ASR-0.6B-hf FS_TREE=$R bash $R/validation_campaigns/speech_q3asr/decode.sh $G $ck earnings_validation1000.jsonl $D/val_$n
done
echo "== grpo + val decodes done $(date)"
