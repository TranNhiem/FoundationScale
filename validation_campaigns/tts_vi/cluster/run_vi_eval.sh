#!/bin/bash
# Vietnamese TTS eval on MiniMax-vi: synthesize with a model, then judge (Qwen3-ASR vi WER + WavLM SIM).
# usage: run_vi_eval.sh GPU MODEL_KIND(voxcpm2) TAG [MODEL_DIR]
set -u
G=$1; K=$2; T=$3; MD=${4:-$HOME/pretraining_weights/Speech-Models/TTS/VoxCPM2}
U=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 1000 ] || { echo "REFUSE gpu $G busy $U"; exit 96; }
E=$HOME/datasets/speech/tts_eval; R=$HOME/Project-Developments/FoundationScale; W=$HOME/pretraining_weights/Speech-Models
CT="env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 -e TORCH_HOME=$HOME/.cache/torch_tts --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute"
mkdir -p $E/synth/$T $E/runs
$CT $HOME/envs/tts-voxcpm/bin/python $R/validation_campaigns/tts_eval/synth_voxcpm2.py --model $MD --testset-dir $E/${VISET:-minimax_vi} --lang vi --list meta.lst --out $E/synth/$T ${LORA:+--lora-ckpt $LORA} < /dev/null > $E/synth/$T.log 2>&1 || { echo "synth failed"; exit 1; }
python3 - "$E" "$T" <<'PY'
import json, os, sys
E, T = sys.argv[1], sys.argv[2]
with open(f"{E}/synth/{T}.judge.jsonl", "w", encoding="utf-8") as o:
    for line in open(f"{E}/{os.environ.get("VISET", "minimax_vi")}/vi/meta.lst", encoding="utf-8").read().splitlines():
        i, pt, pw, tt = line.split("|")[:4]
        o.write(json.dumps({"id": i, "audio": f"{E}/synth/{T}/{i}.wav", "text": tt, "prompt_audio": f"{E}/{os.environ.get("VISET", "minimax_vi")}/vi/{pw}"}, ensure_ascii=False) + "\n")
PY
$CT $HOME/envs/hf-cand-519/bin/python $R/validation_campaigns/tts_eval/judge_multilingual.py --manifest $E/synth/$T.judge.jsonl --asr qwen3asr --language vi \
  --qwen3asr $W/Qwen3-ASR-1.7B-hf --wavlm-ckpt $E/gd/wavlm_large_finetune.pth --seed-code $E/seed-tts-eval-code --python $HOME/envs/tts-judge/bin/python \
  --out $E/runs/vi_$T.json < /dev/null > $E/runs/vi_$T.log 2>&1
echo "exit=$?" >> $E/runs/vi_$T.log
