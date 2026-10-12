#!/bin/bash
# P0.3 parity: verl GRPO mirroring the FS run train100_tp1_<seed> (top_p 1.0, T 1.0, lr 1e-6,
# 4 tasks x 8 samples, 100 steps), inside the GB200 verl container.
#   verl_parity_launch.sh <seed> <gpu> <model_path> <music_jsonl> <abc2midi_bin> <out_dir> <fs_src>
# Every override below is checked against verl 9058412's generated ppo_trainer.yaml.
# FS agentic_grpo (rl/group_policy.py:715-740): centred-only advantage (no std division),
# symmetric 0.2 clip, prompt_mean reduction. ASSUMPTION: verl "seq-mean-token-mean" is the closest
# verl reduction to FS prompt_mean; the definitions are compared in the parity report, not assumed.
set -euo pipefail
SEED=$1; GPU=$2; MODEL=$3; TASKS=$4; ABC=$5; OUT=$6; FS_SRC=$7
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 PYTHONPATH=$FS_SRC VLLM_USE_V1=1
python3 - "$TASKS" "$ABC" "$OUT/music_train.parquet" <<'PY'
import sys, pandas as pd
from foundationscale.agentic_rl.verl_rewards import music_rows_from_jsonl
rows = music_rows_from_jsonl(open(sys.argv[1]), abc2midi_bin=sys.argv[2], timeout_s=60.0)
pd.DataFrame(rows).to_parquet(sys.argv[3]); print("rows", len(rows))
PY
python3 -m verl.trainer.main_ppo \
  data.train_files="['$OUT/music_train.parquet']" data.val_files="['$OUT/music_train.parquet']" \
  data.train_batch_size=4 data.max_prompt_length=1024 data.max_response_length=12000 \
  data.return_raw_chat=True data.seed="$SEED" data.shuffle=True \
  actor_rollout_ref.model.path="$MODEL" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.weight_decay=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.prompt_length=1024 actor_rollout_ref.rollout.response_length=12000 \
  actor_rollout_ref.rollout.seed="$SEED" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  reward.custom_reward_function.path="$FS_SRC/foundationscale/agentic_rl/verl_rewards.py" \
  reward.custom_reward_function.name=music_compute_score \
  reward.reward_model.enable=False \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.total_training_steps=100 \
  trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=10 \
  trainer.default_local_dir="$OUT/checkpoints" trainer.logger='[console]' \
  trainer.project_name=fs_parity trainer.experiment_name="verl_tp1_s$SEED"
