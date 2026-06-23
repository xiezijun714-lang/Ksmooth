#!/usr/bin/env bash
# SAPO | MoE | vLLM rollout | FSDP training | NVIDIA GPUs
# SAPO (Smooth Advantage PO) replaces ratio clipping with a smooth tau-parameterized surrogate (arXiv:2511.20347).

set -xeuo pipefail
export VLLM_USE_V1=1
export HF_HOME=${HF_HOME:-/root/paddlejob/workspace/env_run/xzj/models/.cache/huggingface}
export VERL_GRADIENT_CHECKPOINTING_USE_REENTRANT=${VERL_GRADIENT_CHECKPOINTING_USE_REENTRANT:-0}
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]] || [[ "${PYTORCH_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
fi
export WANDB_MODE=online
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_ENTITY="${WANDB_ENTITY:-515718106-pku}"

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-/root/paddlejob/workspace/env_run/xzj/models/Qwen3-30B-A3B}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

tau_pos=${TAU_POS:-1.0}
tau_neg=${TAU_NEG:-1.05}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-10240}
enable_gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING:-True}
use_torch_compile=${USE_TORCH_COMPILE:-False}
actor_param_offload=${ACTOR_PARAM_OFFLOAD:-False}
actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-True}
ref_param_offload=${REF_PARAM_OFFLOAD:-True}

actor_lr=${ACTOR_LR:-1e-6}
entropy_coeff=${ENTROPY_COEFF:-0}

rollout_tp=${ROLLOUT_TP:-8}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.30}
rollout_n=${ROLLOUT_N:-8}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}
rollout_calculate_log_probs=${ROLLOUT_CALCULATE_LOG_PROBS:-True}

total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-50}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-ksmooth}
experiment_name=${EXPERIMENT_NAME:-qwen3_30b_a3b_sapo_fsdp_aime25}
# ---- end user-adjustable ----

train_file=${TRAIN_FILE:-/root/paddlejob/workspace/env_run/xzj/data/dapo-math-17k/train_dedup.parquet}
val_file=${VAL_FILE:-/root/paddlejob/workspace/env_run/xzj/data/aime-2025/test.parquet}
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$train_file']"
    data.val_files="['$val_file']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing}
)

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=sapo
    actor_rollout_ref.actor.tau_pos=${tau_pos}
    actor_rollout_ref.actor.tau_neg=${tau_neg}
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_torch_compile=${use_torch_compile}
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=${use_torch_compile}
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.calculate_log_probs=${rollout_calculate_log_probs}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.use_torch_compile=${use_torch_compile}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=${use_torch_compile}
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=True
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    +trainer.ray_master_port_range="[39000,40000]"
)

EXTRA=(
)

########################### launch ###########################
python -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
