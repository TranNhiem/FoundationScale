#!/bin/bash
# Waits for T2b training to finish on this tray, then evaluates base + every LoRA checkpoint on
# MiniMax-vi and FLEURS-vi (same GPU). usage: after_t2b.sh GPU
G=$1; D=$HOME/datasets/speech/vi_tts; E=$HOME/datasets/speech/tts_eval
until grep -q '^exit=' $D/logs/t2b_lora.log 2>/dev/null; do sleep 300; done
sleep 30
for set in minimax_vi fleurs_vi_tts; do
  VISET=$set $E/run_vi_eval.sh $G voxcpm2 base_$set
  for ck in $(ls -d $D/checkpoints/t2b_lora/step_* 2>/dev/null | sort); do
    s=$(basename $ck); VISET=$set LORA=$ck $E/run_vi_eval.sh $G voxcpm2 t2b_${s}_$set
  done
done
echo "t2b eval done $(date)" > $D/logs/t2b_eval_done.txt
