#!/bin/bash
# T2b: VoxCPM2 LoRA on the commercial-safe Vietnamese pool; upstream voxcpm_v2 LoRA config with paths filled.
D=$HOME/datasets/speech/vi_tts; C=$D/voxcpm_code/conf/voxcpm_v2/voxcpm_finetune_lora.yaml; O=$D/configs/t2b_lora.yaml
sed -e "s|^pretrained_path: .*|pretrained_path: $HOME/pretraining_weights/Speech-Models/TTS/VoxCPM2/|" \
    -e "s|^train_manifest: .*|train_manifest: $D/voxcpm/train.jsonl|" \
    -e "s|^val_manifest: .*|val_manifest: $D/voxcpm/dev.jsonl|" \
    -e "s|^num_iters: .*|num_iters: 2000|" -e "s|^max_steps: .*|max_steps: 2000|" \
    -e "s|^save_path: .*|save_path: $D/checkpoints/t2b_lora|" \
    -e "s|^tensorboard: .*|tensorboard: $D/logs/t2b_lora|" $C > $O
grep -E "pretrained_path|manifest|max_steps|save_path|learning_rate|batch_size|grad_accum|max_batch_tokens" $O
