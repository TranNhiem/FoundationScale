#!/bin/bash
# T2b full-FT final checkpoint on both Vietnamese sets. usage: run_full_eval.sh GPU
G=$1; D=$HOME/datasets/speech/vi_tts; E=$HOME/datasets/speech/tts_eval
for set in minimax_vi fleurs_vi_tts; do VISET=$set $E/run_vi_eval.sh $G voxcpm2 t2bfull_step2000_$set $D/checkpoints/t2b_full_eval_step2000; done
