#!/bin/bash
# T3 check: does the speaker LoRA keep zero-shot cloning? Seed-TTS test-en with the LoRA, scored by the T0 harness.
set -u
G=${CUDA_VISIBLE_DEVICES:?}; E=$HOME/datasets/speech/tts_eval; T=voxcpm2_t3lora_en
LORA=$HOME/datasets/speech/libritts_r/t3/ckpt/step_0000250 bash $E/run_synth_voxcpm.sh $G en meta.lst $T; tail -1 $E/synth/$T.log
bash $E/run_score.sh $G en meta.lst $T; tail -1 $E/runs/score_$T.log
