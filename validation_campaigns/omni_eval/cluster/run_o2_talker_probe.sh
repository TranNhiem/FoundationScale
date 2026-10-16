#!/bin/bash
# O2 re-scope probe: Qwen3-Omni talker read-aloud, English control (Seed-TTS test-en) vs Vietnamese (FLEURS-vi), 200 sentences each.
# usage: gpu-pool run 1 --job foxbrain-o2-talker-probe -- bash run_o2_talker_probe.sh
set -u
G=${CUDA_VISIBLE_DEVICES:?run through gpu-pool or set CUDA_VISIBLE_DEVICES}
E=$HOME/datasets/speech/tts_eval; R=$HOME/Project-Developments/FoundationScale; W=$HOME/pretraining_weights/Speech-Models; O=$E/o2_talker
CT="env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 -e TORCH_HOME=$HOME/.cache/torch_tts --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute"
PY=$HOME/envs/hf-cand-519/bin/python; mkdir -p $O
for L in en vi; do
  META=$E/seedtts_testset/en/meta.lst; [ $L = vi ] && META=$E/fleurs_vi_tts/vi/meta.lst
  echo "== synth $L $(date)"
  $CT $PY $R/validation_campaigns/omni_eval/synth_qwen3omni_talker.py --model $W/Qwen3-Omni-30B-A3B-Instruct --meta $META --lang $L --out $O/$L < /dev/null > $O/synth_$L.log 2>&1 || { echo "synth $L failed"; tail -5 $O/synth_$L.log; continue; }
  echo "== judge $L $(date)"
  if [ $L = en ]; then
    $CT $HOME/envs/tts-judge/bin/python $R/validation_campaigns/tts_eval/judge_multilingual.py --manifest $O/en/judge.jsonl --asr whisper --language en --whisper $W/whisper-large-v3 --no-sim --out $O/judge_en.json < /dev/null > $O/judge_en.log 2>&1
  else
    $CT $PY $R/validation_campaigns/tts_eval/judge_multilingual.py --manifest $O/vi/judge.jsonl --asr qwen3asr --language vi --qwen3asr $W/Qwen3-ASR-1.7B-hf --no-sim --out $O/judge_vi.json < /dev/null > $O/judge_vi.log 2>&1
  fi
  echo "judge $L exit=$?"
done
echo "== done $(date)"
