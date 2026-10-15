#!/bin/bash
# T2a: Qwen3-TTS-1.7B single-speaker SFT on LibriTTS-R speaker 6209, then held-out eval. usage: run_t2a.sh GPU
set -u
G=$1; D=$HOME/datasets/speech/libritts_r; T=$D/t2a; W=$HOME/pretraining_weights/Speech-Models; R=$HOME/Project-Developments/FoundationScale; E=$HOME/datasets/speech/tts_eval
U=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 1000 ] || { echo "REFUSE gpu $G busy $U"; exit 96; }
CT="env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 -e TORCH_HOME=$HOME/.cache/torch_tts --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute"
PY=$HOME/envs/tts-qwen/bin/python; FT=$D/qwen3tts_code/finetuning
step() { echo "== $1 $(date)"; }
[ -s $T/train_with_codes.jsonl ] || { step prepare; $CT bash -c "cd $FT && $PY prepare_data.py --device cuda:0 --tokenizer_model_path $W/TTS/Qwen3-TTS-Tokenizer-12Hz --input_jsonl $T/train_raw.jsonl --output_jsonl $T/train_with_codes.jsonl" < /dev/null || exit 1; }
step sft; $CT bash -c "cd $FT && $PY $R/validation_campaigns/tts_train/qwen3tts_sft.py --upstream-script $FT/sft_12hz.py -- --init_model_path $W/TTS/Qwen3-TTS-12Hz-1.7B-Base --output_model_path $T/ckpt --train_jsonl $T/train_with_codes.jsonl --batch_size 32 --lr 2e-6 --num_epochs 10 --speaker_name spk6209" < /dev/null || exit 1
CK=$(ls -d $T/ckpt/checkpoint-epoch-* | sort -V | tail -1); echo "final checkpoint $CK"
for m in clone custom; do
  MD=$W/TTS/Qwen3-TTS-12Hz-1.7B-Base; [ $m = custom ] && MD=$CK
  step synth_$m; $CT $PY $R/validation_campaigns/tts_train/synth_qwen3tts_heldout.py --model $MD --heldout $T/heldout.jsonl --mode $m --speaker spk6209 --out $T/eval_$m < /dev/null || exit 1
  step judge_$m; $CT $HOME/envs/tts-judge/bin/python $R/validation_campaigns/tts_eval/judge_multilingual.py --manifest $T/eval_$m/judge.jsonl --asr whisper --language en \
    --whisper $W/whisper-large-v3 --wavlm-ckpt $E/gd/wavlm_large_finetune.pth --seed-code $E/seed-tts-eval-code --python $HOME/envs/tts-judge/bin/python --out $T/eval_$m.json < /dev/null
done
step done
