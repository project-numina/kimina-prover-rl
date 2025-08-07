# This config is intended for debugging purposes. 

set -x

WORKING_DIRECTORY=.
PROMPT_SET_TRAIN=AI-MO/Kimina-Prover-Promptset
PROMPT_SET_TEST=AI-MO/minif2f_test
DATA_SAMPLE_SIZE=8

max_prompt_length=1684
model_max_len=8192
max_response_length=$((model_max_len-max_prompt_length))
actor_ppo_max_token_len=$model_max_len
infer_ppo_max_token_len=$model_max_len

python3 $WORKING_DIRECTORY/prepare_data.py \
  --train-dataset $PROMPT_SET_TRAIN \
  --test-dataset $PROMPT_SET_TEST \
  --path $WORKING_DIRECTORY \
  --sample-first-n $DATA_SAMPLE_SIZE

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.train_files="[$WORKING_DIRECTORY/prompt_sets/$PROMPT_SET_TRAIN-first-$DATA_SAMPLE_SIZE/train.parquet]" \
    data.val_files="[$WORKING_DIRECTORY/prompt_sets/$PROMPT_SET_TRAIN-first-$DATA_SAMPLE_SIZE/test.parquet]" \
    data.train_batch_size=8 \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    +data.return_extra_info=True \
    data.return_raw_chat=True \
    data.dataloader_num_workers=0 \
    data.custom_cls.path="${WORKING_DIRECTORY}/kimina_prover_rl/dataset.py" \
    data.custom_cls.name=NuminaRLDataset \
    data.truncation='error' \
    actor_rollout_ref.model.path=AI-MO/Kimina-Prover-Distill-1.7B \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.max_num_batched_tokens="${infer_ppo_max_token_len}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    reward_model.launch_reward_fn_async=True \
    custom_reward_function.path="${WORKING_DIRECTORY}/kimina_prover_rl/reward/reward.py" \
    custom_reward_function.name=reward \
    +custom_reward_function.reward_kwargs.return_dict=True \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='kimina-prover' \
    trainer.experiment_name='kimina-prover-1.7B-dry-run' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
