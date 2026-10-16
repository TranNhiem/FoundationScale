#!/bin/bash
# SP-6 gate: Qwen3-TTS-12Hz tokenizer codes decoded by its own decoder vs Qwen3-Omni Code2Wav, 100 FLEURS-vi recordings.
# usage: gpu-pool run 1 --job foxbrain-sp6-codec -- bash run_codec_roundtrip.sh   (FS_TREE = FoundationScale tree)
set -u
G=${CUDA_VISIBLE_DEVICES:?run through gpu-pool}; R=${FS_TREE:?}
E=$HOME/datasets/speech/tts_eval; W=$HOME/pretraining_weights/Speech-Models; O=$E/sp6_codec; M=$E/fleurs_vi_test_gt.jsonl
CT="env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute"
S=$R/validation_campaigns/omni_eval/codec_roundtrip.py; mkdir -p $O
$CT $HOME/envs/tts-qwen/bin/python $S encode --manifest $M --out $O --tokenizer $W/TTS/Qwen3-TTS-Tokenizer-12Hz < /dev/null || { echo "encode failed"; exit 1; }
$CT $HOME/envs/hf-cand-519/bin/python $S omni-decode --manifest $M --out $O --omni $W/Qwen3-Omni-30B-A3B-Instruct < /dev/null || { echo "omni-decode failed"; exit 1; }
for arm in gt tokdec omnidec; do
  $CT $HOME/envs/hf-cand-519/bin/python $R/validation_campaigns/tts_eval/judge_multilingual.py --manifest $O/$arm.judge.jsonl --asr qwen3asr --language vi \
    --qwen3asr $W/Qwen3-ASR-1.7B-hf --no-sim --out $O/judge_$arm.json < /dev/null > $O/judge_$arm.log 2>&1; echo "judge $arm exit=$?"
done
echo "== done $(date)"
