export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1,2,3

PROJECT_NAME=chatkbqa

EXP_NAME=chatkbqa_7b
# INIT_MODEL=meta-llama/Llama-2-7b-hf
INIT_MODEL=meta-llama/Llama-3.2-3B-Instruct
# ADAPTER=./sft_output/checkpoint-3500
SFT_MODEL=./sft_merged_checkpoint

DATE=$(date '+%Y-%m-%d-%H-%M-%S')

python3 -m verl.trainer.CUSTOM_main_ppo \
    data.train_files=data/ChatKBQA/WebQSP/train.parquet \
    data.val_files=data/ChatKBQA/WebQSP/test_full.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=1260 \
    data.max_response_length=256 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    critic.ppo_micro_batch_size=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.optim.lr=1e-5 \
    critic.model.enable_gradient_checkpointing=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console'] \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=20 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    actor_rollout_ref.model.path=$SFT_MODEL \
    critic.model.path=$INIT_MODEL \
    trainer.default_local_dir=/dev/ana/training_outputs/${EXP_NAME} \
    trainer.total_epochs=5 2>&1 | tee exp_log/$PROJECT_NAME-7b-ppo-verl_demo.log 


# trainer.logger=['wandb'] \
# tee exp_log/$PROJECT_NAME-7b-ppo-verl_demo_$DATE.log 