#!/bin/bash
# T3: VoxCPM2 LoRA speaker adaptation on the T2a data (LibriTTS-R speaker 6209, 398 train / 60 held out), mirroring T2a:
# 10 epochs (250 steps at effective batch 16), final step fixed in advance; then held-out eval of base clone vs LoRA clone.
# usage (via the GPU pool, which sets CUDA_VISIBLE_DEVICES): gpu-pool run 1 --job foxbrain-t3-voxcpm-spk -- bash run_t3.sh
set -u
G=${CUDA_VISIBLE_DEVICES:?run through gpu-pool or set CUDA_VISIBLE_DEVICES}
D=$HOME/datasets/speech/libritts_r; T=$D/t2a; O=$D/t3; V=$HOME/datasets/speech/vi_tts/voxcpm_code
W=$HOME/pretraining_weights/Speech-Models; R=$HOME/Project-Developments/FoundationScale; E=$HOME/datasets/speech/tts_eval
CT="env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 -e TORCH_HOME=$HOME/.cache/torch_tts --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute"
PY=$HOME/envs/tts-voxcpm/bin/python
step() { echo "== $1 $(date)"; }
mkdir -p $O/wav16 $O/testset/en $O/logs
step data
cat > $O/build_t3.py <<'PY'
import json, sys
from math import gcd
from pathlib import Path
import soundfile as sf
from scipy.signal import resample_poly
T, O = Path(sys.argv[1]), Path(sys.argv[2])
SR = 16000  # VoxCPM2 AudioVAE encoder rate (upstream voxcpm_v2 config asserts it)
def to16(src):
    dst = O / "wav16" / Path(src).name
    if not dst.exists():
        data, sr = sf.read(src, dtype="float32")
        g = gcd(sr, SR)
        sf.write(dst, resample_poly(data, SR // g, sr // g), SR, subtype="PCM_16")
    return dst, sf.info(dst).duration
for split, src in (("train", "train_raw.jsonl"), ("dev", "heldout.jsonl")):
    with (O / f"{split}.jsonl").open("w") as f:
        for line in (T / src).read_text().splitlines():
            r = json.loads(line)
            p, d = to16(r["audio"])
            f.write(json.dumps({"audio": str(p), "text": r["text"], "duration": round(d, 3), "dataset_id": 0}) + "\n")
held = [json.loads(x) for x in (T / "heldout.jsonl").read_text().splitlines()]
with (O / "testset/en/meta.lst").open("w") as f:
    for r in held:  # clone from the same fixed reference clip T2a used
        f.write(f"{r['id']}|{r['ref_text']}|{r['ref_audio']}|{r['text']}\n")
with (O / "judge_rows.jsonl").open("w") as f:  # SIM against the real held-out recording, as in T2a
    for r in held:
        f.write(json.dumps({"id": r["id"], "text": r["text"], "prompt_audio": r["audio"]}) + "\n")
PY
$CT $PY $O/build_t3.py $T $O < /dev/null || { echo "data failed"; exit 1; }
sed -e "s|^pretrained_path: .*|pretrained_path: $W/TTS/VoxCPM2/|" -e "s|^train_manifest: .*|train_manifest: $O/train.jsonl|" \
    -e "s|^val_manifest: .*|val_manifest: $O/dev.jsonl|" -e "s|^num_iters: .*|num_iters: 250|" -e "s|^max_steps: .*|max_steps: 250|" \
    -e "s|^warmup_steps: .*|warmup_steps: 25|" -e "s|^save_interval: .*|save_interval: 250|" -e "s|^valid_interval: .*|valid_interval: 250|" \
    -e "s|^save_path: .*|save_path: $O/ckpt|" -e "s|^tensorboard: .*|tensorboard: $O/logs/tb|" $V/conf/voxcpm_v2/voxcpm_finetune_lora.yaml > $O/t3_lora.yaml
grep -E "max_steps|save_interval|learning_rate|batch_size|grad_accum" $O/t3_lora.yaml
[ -d $O/ckpt/step_0000250 ] || { step train; $CT bash -c "cd $V && $PY -m torch.distributed.run --standalone --nproc_per_node 1 scripts/train_voxcpm_finetune.py --config_path $O/t3_lora.yaml" < /dev/null > $O/logs/train.log 2>&1 || { echo "train failed"; exit 1; }; }
for m in base lora; do
  step synth_$m; $CT $PY $R/validation_campaigns/tts_eval/synth_voxcpm2.py --model $W/TTS/VoxCPM2 --testset-dir $O/testset --lang en --out $O/eval_$m \
    $([ $m = lora ] && echo --lora-ckpt $O/ckpt/step_0000250) < /dev/null > $O/logs/synth_$m.log 2>&1 || { echo "synth $m failed"; exit 1; }
  python3 -c "
import json,sys
with open('$O/eval_$m/judge.jsonl','w') as f:
    for l in open('$O/judge_rows.jsonl'):
        r=json.loads(l); r['audio']='$O/eval_$m/'+r['id']+'.wav'; f.write(json.dumps(r)+'\n')"
  step judge_$m; $CT $HOME/envs/tts-judge/bin/python $R/validation_campaigns/tts_eval/judge_multilingual.py --manifest $O/eval_$m/judge.jsonl --asr whisper --language en \
    --whisper $W/whisper-large-v3 --wavlm-ckpt $E/gd/wavlm_large_finetune.pth --seed-code $E/seed-tts-eval-code --python $HOME/envs/tts-judge/bin/python --out $O/eval_$m.json < /dev/null
done
step done
