#!/usr/bin/env bash
# your-01：按 05 带学时的决定配置（4×A800，Qwen3-8B-Base，LeetCode 去污染版 1889 题，6144 / n=8 / 200 步，mini-batch 4，prime_plus，熵 0）。
set -euo pipefail
cd /root/verl
export CODER1_EXEC=firejail
export CODER1_PLUS_SANDBOX=firejail
export CODER1_PLUS_TMP=/data/tmp/judge
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export RUN=your-01
mkdir -p /data/ckpt/$RUN /data/rollouts/$RUN /data/tmp/judge logs/$RUN
set -x
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/code-r1-plus-2k-leetcode2k-dedup/train.parquet \
    data.val_files=/data/code-r1-plus-2k-leetcode2k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=6144 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    actor_rollout_ref.model.path=/root/models/Qwen3-8B-Base \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    reward_model.reward_manager=prime_plus \
    trainer.logger=[console,wandb] \
    trainer.project_name=code-r1-qwen3 \
    trainer.experiment_name=your-01-qwen3-8b-base-lc2k \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=16 \
    trainer.save_freq=64 \
    trainer.total_epochs=2 \
    trainer.total_training_steps=200 \
    trainer.rollout_data_dir=/data/rollouts/your-01 \
    trainer.default_local_dir=/data/ckpt/your-01 \
    "$@" 2>&1 | tee logs/$RUN/grpo.log
