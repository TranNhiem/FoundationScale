#!/bin/bash
# T1 hard case: Seed-TTS zh/hardcase.lst (400) with VoxCPM2 and Qwen3-TTS-1.7B, synthesize + score on one GPU.
G=$1; E=$HOME/datasets/speech/tts_eval
$E/run_synth_voxcpm.sh $G zh hardcase.lst voxcpm2_hard && $E/run_score.sh $G zh hardcase.lst voxcpm2_hard
$E/run_synth_qwen.sh $G Qwen3-TTS-12Hz-1.7B-Base zh hardcase.lst q17_hard && $E/run_score.sh $G zh hardcase.lst q17_hard
echo "hard chain done $(date)" >> $E/hard_chain.log
