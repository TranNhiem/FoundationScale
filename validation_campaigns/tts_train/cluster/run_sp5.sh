#!/bin/bash
# SP-5: multi-speaker VoxCPM2 LoRA on 8 LibriTTS-R speakers (20 held out each), 3 epochs, final step fixed in advance.
# Eval: held-out clone from each speaker's fixed prompt (base vs LoRA, SIM against the real recording), then the
# zero-shot check on Seed-TTS test-en with the LoRA loaded (vs the T1 base run).
# usage: gpu-pool run 1 --job foxbrain-sp5-voxcpm-multispk -- bash run_sp5.sh   (FS_TREE = FoundationScale tree)
set -u
G=${CUDA_VISIBLE_DEVICES:?run through gpu-pool}; R=${FS_TREE:?}
D=$HOME/datasets/speech/libritts_r; O=$D/sp5; V=$HOME/datasets/speech/vi_tts/voxcpm_code
W=$HOME/pretraining_weights/Speech-Models; E=$HOME/datasets/speech/tts_eval; H=$R/validation_campaigns/tts_train/cluster
CT="env ENROOT_DATA_PATH=$HOME/.local/share/enroot enroot start -e PYTHONNOUSERSITE=1 -e CUDA_VISIBLE_DEVICES=$G -e HF_HUB_OFFLINE=1 -e TORCH_HOME=$HOME/.cache/torch_tts --mount $HOME:$HOME fs-g4e4b-nemo-automodel-26-04_compute"
PY=$HOME/envs/tts-voxcpm/bin/python
step() { echo "== $1 $(date)"; }
mkdir -p $O/logs
[ -s $O/train.jsonl ] || { step data; $CT $PY $H/build_sp5.py $D $O 8 20 < /dev/null || { echo "data failed"; exit 1; }; }
N=$(wc -l < $O/train.jsonl); S=$(( (3 * N + 15) / 16 )); CK=$O/ckpt/$(printf "step_%07d" $S); echo "train rows $N -> $S steps"
sed -e "s|^pretrained_path: .*|pretrained_path: $W/TTS/VoxCPM2/|" -e "s|^train_manifest: .*|train_manifest: $O/train.jsonl|" \
    -e "s|^val_manifest: .*|val_manifest: $O/dev.jsonl|" -e "s|^num_iters: .*|num_iters: $S|" -e "s|^max_steps: .*|max_steps: $S|" \
    -e "s|^warmup_steps: .*|warmup_steps: $((S / 10))|" -e "s|^save_interval: .*|save_interval: $S|" -e "s|^valid_interval: .*|valid_interval: $S|" \
    -e "s|^save_path: .*|save_path: $O/ckpt|" -e "s|^tensorboard: .*|tensorboard: $O/logs/tb|" $V/conf/voxcpm_v2/voxcpm_finetune_lora.yaml > $O/sp5_lora.yaml
[ -d $CK ] || { step train; $CT bash -c "cd $V && $PY -m torch.distributed.run --standalone --nproc_per_node 1 scripts/train_voxcpm_finetune.py --config_path $O/sp5_lora.yaml" < /dev/null > $O/logs/train.log 2>&1 || { echo "train failed"; exit 1; }; }
for m in base lora; do
  step synth_$m; $CT $PY $R/validation_campaigns/tts_eval/synth_voxcpm2.py --model $W/TTS/VoxCPM2 --testset-dir $O/testset --lang en --out $O/eval_$m \
    $([ $m = lora ] && echo --lora-ckpt $CK) < /dev/null > $O/logs/synth_$m.log 2>&1 || { echo "synth $m failed"; exit 1; }
  python3 -c "
import json
with open('$O/eval_$m/judge.jsonl','w') as f:
    for l in open('$O/judge_rows.jsonl'):
        r=json.loads(l); r['audio']='$O/eval_$m/'+r['id']+'.wav'; f.write(json.dumps(r)+'\n')"
  step judge_$m; $CT $HOME/envs/tts-judge/bin/python $R/validation_campaigns/tts_eval/judge_multilingual.py --manifest $O/eval_$m/judge.jsonl --asr whisper --language en \
    --whisper $W/whisper-large-v3 --wavlm-ckpt $E/gd/wavlm_large_finetune.pth --seed-code $E/seed-tts-eval-code --python $HOME/envs/tts-judge/bin/python --out $O/eval_$m.json < /dev/null
done
step zeroshot; LORA=$CK bash $E/run_synth_voxcpm.sh $G en meta.lst voxcpm2_sp5lora_en; tail -1 $E/synth/voxcpm2_sp5lora_en.log
bash $E/run_score.sh $G en meta.lst voxcpm2_sp5lora_en; tail -1 $E/runs/score_voxcpm2_sp5lora_en.log
step done
