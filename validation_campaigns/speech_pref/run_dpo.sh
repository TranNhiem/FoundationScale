#!/bin/bash
# SP-2 stage 2: DPO from the SP-4 0.6B SFT model on the stage-1 pairs, 1 epoch, a checkpoint every quarter.
# Each checkpoint is decoded on earnings_validation1000; the rule fixed in advance picks the lowest FS-normalizer WER
# (the SFT model, 14.09, is the bar), and only that checkpoint is decoded on the held-out test (select_and_test.sh).
# usage: gpu-pool run 1 --job foxbrain-sp2-dpo -- bash run_dpo.sh   (FS_TREE = FoundationScale tree)
set -u
# SFT dir = checkpoint-250 weights + the 0.6B processor files (checkpoints carry weights only)
G=${CUDA_VISIBLE_DEVICES:?run through gpu-pool}; R=${FS_TREE:?}
S=$HOME/datasets/speech; W=$HOME/pretraining_weights/Speech-Models; O=$S/sp2_pref; SFT=$S/sp4_q3asr06/sft250_with_processor
CT="env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e PYTHONPATH=$R/src -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute"
P=${PAIRS_PER_STEP:-16}; N=$(wc -l < $O/pairs.jsonl); STEPS=$(( ${EPOCHS:-1} * N / P )); EVERY=$(( STEPS / 4 )); D=$O/dpo${RUN:+_$RUN}
echo "pairs $N -> $STEPS steps, save every $EVERY"
[ -d $D/final ] || $CT $HOME/envs/hf-cand-519/bin/python $R/validation_campaigns/speech_pref/train_dpo.py --model $SFT --pairs $O/pairs.jsonl \
  --save-dir $D --beta ${BETA:-0.1} --sft-weight ${SFTW:-0.0} --learning-rate ${LR:-1e-6} --pairs-per-step $P --max-steps $STEPS --save-every $EVERY --seed ${SEED:-0} < /dev/null > $D.log 2>&1 || { echo "dpo failed"; tail -5 $D.log; exit 1; }
for ck in $(ls -d $D/step_* $D/final 2>/dev/null); do
  n=$(basename $ck); [ -s $D/val_$n/eval.json ] || PROC=$W/Qwen3-ASR-0.6B-hf FS_TREE=$R bash $R/validation_campaigns/speech_q3asr/decode.sh $G $ck earnings_validation1000.jsonl $D/val_$n
done
echo "== dpo + val decodes done $(date)"
