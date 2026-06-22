#!/usr/bin/env bash
# BrowseComp-Plus GRPO | Qwen3-30B-A3B | fully async | Megatron train + SGLang rollout
# Default layout: 4 nodes x 8 GPUs, 2 trainer nodes + 2 rollout nodes.

set -xeuo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/paddlejob/workspace/env_run/xzj/Ksmooth}
PYTHON_BIN=${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}
RAY_BIN=${RAY_BIN:-${PROJECT_DIR}/.venv/bin/ray}
SSH_PORT=${SSH_PORT:-42701}

TRAIN_NNODES=${TRAIN_NNODES:-2}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-2}
DEFAULT_TRAINER_IPS=${DEFAULT_TRAINER_IPS:-10.52.100.77,10.52.107.89,10.52.100.83,10.52.104.22}
select_cluster_nodes() {
    if [[ -n "${TRAIN_WORKER_IPS:-}" || -n "${ROLLOUT_WORKER_IPS:-}" || -n "${WORKER_IPS:-}" ]]; then
        HEAD_IP=${HEAD_IP:-${POD_IP:-10.52.100.77}}
        TRAIN_WORKER_IPS=${TRAIN_WORKER_IPS:-"10.52.100.83"}
        ROLLOUT_WORKER_IPS=${ROLLOUT_WORKER_IPS:-"10.52.104.22"}
        WORKER_IPS=${WORKER_IPS:-"${TRAIN_WORKER_IPS} ${ROLLOUT_WORKER_IPS}"}
        return
    fi

    local node_source="${TRAINER_IPS:-${DEFAULT_TRAINER_IPS:-${PADDLE_TRAINERS:-}}}"
    if [[ -z "${node_source}" ]]; then
        HEAD_IP=${HEAD_IP:-${POD_IP:-10.52.100.77}}
        TRAIN_WORKER_IPS="10.52.100.83"
        ROLLOUT_WORKER_IPS="10.52.104.22"
        WORKER_IPS="${TRAIN_WORKER_IPS} ${ROLLOUT_WORKER_IPS}"
        return
    fi

    IFS=',' read -r -a all_nodes <<< "${node_source}"
    HEAD_IP=${HEAD_IP:-${POD_IP:-${all_nodes[0]}}}
    local selected_nodes=("${HEAD_IP}")
    local ip
    for ip in "${all_nodes[@]}"; do
        [[ -z "${ip}" || "${ip}" == "${HEAD_IP}" ]] && continue
        selected_nodes+=("${ip}")
    done

    local needed_nodes=$((TRAIN_NNODES + ROLLOUT_NNODES))
    if [[ "${#selected_nodes[@]}" -lt "${needed_nodes}" ]]; then
        echo "[config] need ${needed_nodes} nodes but only selected ${#selected_nodes[@]} from ${node_source}" >&2
        exit 1
    fi

    TRAIN_WORKER_IPS="${selected_nodes[*]:1:TRAIN_NNODES-1}"
    ROLLOUT_WORKER_IPS="${selected_nodes[*]:TRAIN_NNODES:ROLLOUT_NNODES}"
    WORKER_IPS="${TRAIN_WORKER_IPS} ${ROLLOUT_WORKER_IPS}"
}

select_cluster_nodes
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/dev/shm/ray_tmp/ray}
START_RAY_CLUSTER=${START_RAY_CLUSTER:-1}
START_RETRIEVER=${START_RETRIEVER:-1}
WAIT_RETRIEVER=${WAIT_RETRIEVER:-1}
RETRIEVER_HOST=${RETRIEVER_HOST:-${HEAD_IP}}
RETRIEVER_PORT=${RETRIEVER_PORT:-8000}

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HF_HOME=${HF_HOME:-/root/paddlejob/workspace/env_run/xzj/models/.cache/huggingface}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-${PROJECT_DIR}}
export ROLLOUT_MASTER_PORT_RANGE=${ROLLOUT_MASTER_PORT_RANGE:-41000,42000}
export SGLANG_SERVER_PORT_RANGE=${SGLANG_SERVER_PORT_RANGE:-30000,39000}
export SGLANG_SERVER_PORT_STRIDE=${SGLANG_SERVER_PORT_STRIDE:-512}
export VERL_DATAPROTO_SERIALIZATION_METHOD=${VERL_DATAPROTO_SERIALIZATION_METHOD:-numpy}
export VERL_GRADIENT_CHECKPOINTING_USE_REENTRANT=${VERL_GRADIENT_CHECKPOINTING_USE_REENTRANT:-0}
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]] || [[ "${PYTORCH_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
fi

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_ENTITY=${WANDB_ENTITY:-515718106-pku}
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-3881bb8218444692be4aa4ea02f8dfdb}"
export ONEAPI_KEY="${ONEAPI_KEY:-sk-vtZWLiyN5qzCPE2176CeEf5963B547E8814eB7D7BbD258B3}"
export BCP_JUDGE_API_KEY_ENV=${BCP_JUDGE_API_KEY_ENV:-ONEAPI_KEY}
export BCP_JUDGE_API_BASE=${BCP_JUDGE_API_BASE:-https://oneapi-comate.baidu-int.com/v1}
export BCP_JUDGE_MODEL=${BCP_JUDGE_MODEL:-Deepseek-V4-Flash}
if [[ -z "${!BCP_JUDGE_API_KEY_ENV:-}" ]]; then
    echo "[config] required judge API key env ${BCP_JUDGE_API_KEY_ENV} is not set." >&2
    echo "[config] export ${BCP_JUDGE_API_KEY_ENV}=<your-key> before launching BCP training." >&2
    exit 1
fi
if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "[config] WANDB_MODE=online but WANDB_API_KEY is not set." >&2
    echo "[config] export WANDB_API_KEY=<your-key>, or set WANDB_MODE=offline/disabled." >&2
    exit 1
fi

start_ray_cluster() {
    for worker_ip in ${WORKER_IPS}; do
        ssh -p "${SSH_PORT}" -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${worker_ip}" true
    done

    "${RAY_BIN}" stop --force || true
    mkdir -p "${RAY_TEMP_DIR}"
    "${RAY_BIN}" start \
        --head \
        --node-ip-address="${HEAD_IP}" \
        --port="${RAY_PORT}" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT}" \
        --temp-dir="${RAY_TEMP_DIR}" \
        --num-gpus="${TRAIN_GPUS}" \
        --resources='{"train_node": 1}' \
        --disable-usage-stats

    for worker_ip in ${WORKER_IPS}; do
        ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no "${worker_ip}" \
            "cd '${PROJECT_DIR}' && '${RAY_BIN}' stop --force || true"
    done

    for worker_ip in ${TRAIN_WORKER_IPS}; do
        ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no "${worker_ip}" \
            "export CUDA_DEVICE_MAX_CONNECTIONS='${CUDA_DEVICE_MAX_CONNECTIONS}' FLASHINFER_WORKSPACE_BASE='${FLASHINFER_WORKSPACE_BASE}' ROLLOUT_MASTER_PORT_RANGE='${ROLLOUT_MASTER_PORT_RANGE}' SGLANG_SERVER_PORT_RANGE='${SGLANG_SERVER_PORT_RANGE}' SGLANG_SERVER_PORT_STRIDE='${SGLANG_SERVER_PORT_STRIDE}' && mkdir -p '${RAY_TEMP_DIR}' && cd '${PROJECT_DIR}' && '${RAY_BIN}' start --address='${HEAD_IP}:${RAY_PORT}' --node-ip-address='${worker_ip}' --temp-dir='${RAY_TEMP_DIR}' --num-gpus='${TRAIN_GPUS}' --resources='{\"train_node\": 1}' --disable-usage-stats"
    done

    for worker_ip in ${ROLLOUT_WORKER_IPS}; do
        ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no "${worker_ip}" \
            "export CUDA_DEVICE_MAX_CONNECTIONS='${CUDA_DEVICE_MAX_CONNECTIONS}' FLASHINFER_WORKSPACE_BASE='${FLASHINFER_WORKSPACE_BASE}' ROLLOUT_MASTER_PORT_RANGE='${ROLLOUT_MASTER_PORT_RANGE}' SGLANG_SERVER_PORT_RANGE='${SGLANG_SERVER_PORT_RANGE}' SGLANG_SERVER_PORT_STRIDE='${SGLANG_SERVER_PORT_STRIDE}' && mkdir -p '${RAY_TEMP_DIR}' && cd '${PROJECT_DIR}' && '${RAY_BIN}' start --address='${HEAD_IP}:${RAY_PORT}' --node-ip-address='${worker_ip}' --temp-dir='${RAY_TEMP_DIR}' --num-gpus='${ROLLOUT_GPUS}' --resources='{\"rollout_node\": 1}' --disable-usage-stats"
    done

    "${RAY_BIN}" status --address="${HEAD_IP}:${RAY_PORT}"
}

# ---- paths ----
MODEL_PATH=${MODEL_PATH:-/root/paddlejob/workspace/env_run/xzj/models/Qwen3-30B-A3B}
MCORE_MODEL_PATH=${MCORE_MODEL_PATH:-}
DATA_DIR=${DATA_DIR:-/root/paddlejob/workspace/env_run/xzj/dataset/browsecomp-plus-processed}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.paper.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/test.paper.parquet}
RETRIEVER_MODEL_PATH=${RETRIEVER_MODEL_PATH:-/root/paddlejob/workspace/env_run/xzj/models/Qwen3-Embedding-8B}
RETRIEVER_CORPUS_FILE=${RETRIEVER_CORPUS_FILE:-${DATA_DIR}/corpus.parquet}
RETRIEVER_DENSE_CACHE=${RETRIEVER_DENSE_CACHE:-/root/paddlejob/workspace/env_run/xzj/browsecomp_dense_cache_tevatron.pkl}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-30b-a3b-bcp-grpo-fully-async-4node-32k}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/logs}
CKPT_DIR=${CKPT_DIR:-${PROJECT_DIR}/ckpt/${EXPERIMENT_NAME}}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${PROJECT_DIR}/rollout_outputs/${EXPERIMENT_NAME}}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${PROJECT_DIR}/val_outputs/${EXPERIMENT_NAME}}
mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${ROLLOUT_DATA_DIR}" "${VALIDATION_DATA_DIR}"

for required_path in "${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}" "${RETRIEVER_MODEL_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[config] required path missing: ${required_path}" >&2
        exit 1
    fi
done
if [[ ! -e "${RETRIEVER_DENSE_CACHE}" && ! -e "${RETRIEVER_CORPUS_FILE}" ]]; then
    echo "[config] retriever needs dense cache or corpus file." >&2
    exit 1
fi

# ---- cluster shape ----
TRAIN_NNODES=${TRAIN_NNODES:-2}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-2}
TRAIN_GPUS=${TRAIN_GPUS:-8}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}

# ---- training knobs ----
train_batch_size=${TRAIN_BATCH_SIZE:-32}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-4}
n_resp=${N_RESP:-8}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-32768}
max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH:-16000}
max_model_len=${MAX_MODEL_LEN:-40960}
model_context_limit=${MODEL_CONTEXT_LIMIT:-40960}

actor_tp=${ACTOR_TP:-2}
actor_pp=${ACTOR_PP:-1}
actor_cp=${ACTOR_CP:-4}
actor_ep=${ACTOR_EP:-8}
actor_etp=${ACTOR_ETP:-1}
ref_tp=${REF_TP:-2}
ref_pp=${REF_PP:-1}
ref_cp=${REF_CP:-4}
ref_ep=${REF_EP:-${actor_ep}}
ref_etp=${REF_ETP:-${actor_etp}}
rollout_tp=${ROLLOUT_TP:-2}

rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.70}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-16}
rollout_quantization=${ROLLOUT_QUANTIZATION:-fp8}
rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}
update_weights_bucket_mb=${UPDATE_WEIGHTS_BUCKET_MB:-1600}
sglang_disable_overlap_schedule=${SGLANG_DISABLE_OVERLAP_SCHEDULE:-True}
sglang_chunked_prefill_size=${SGLANG_CHUNKED_PREFILL_SIZE:-8192}
sglang_max_prefill_tokens=${SGLANG_MAX_PREFILL_TOKENS:-32768}

total_rollout_steps=${TOTAL_ROLLOUT_STEPS:-1000000000}
total_training_steps=${TOTAL_TRAINING_STEPS:-$((total_rollout_steps / ppo_mini_batch_size))}
if [[ "${total_training_steps}" -lt 1 ]]; then
    total_training_steps=1
fi
require_batches=${REQUIRE_BATCHES:-1}
staleness_threshold=${STALENESS_THRESHOLD:-0.5}
trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-$((train_batch_size / ppo_mini_batch_size))}
if [[ "${trigger_parameter_sync_step}" -lt 1 ]]; then
    trigger_parameter_sync_step=1
fi
partial_rollout=${PARTIAL_ROLLOUT:-True}
drain_before_sync=${DRAIN_BEFORE_SYNC:-False}
rollout_correction_bypass_mode=${ROLLOUT_CORRECTION_BYPASS_MODE:-False}
fully_async_skip_old_logprob_cpu_snapshot=${FULLY_ASYNC_SKIP_OLD_LOGPROB_CPU_SNAPSHOT:-0}
megatron_cpu_snapshot_pin_memory=${VERL_MEGATRON_CPU_SNAPSHOT_PIN_MEMORY:-0}

token_budget=$((max_prompt_length + max_response_length))
if [[ "${token_budget}" -ge "${max_model_len}" ]]; then
    echo "[config] MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH=${token_budget} must be < MAX_MODEL_LEN=${max_model_len}" >&2
    exit 1
fi
if [[ "${max_model_len}" -gt "${model_context_limit}" ]]; then
    echo "[config] MAX_MODEL_LEN=${max_model_len} exceeds MODEL_CONTEXT_LIMIT=${model_context_limit}" >&2
    exit 1
fi
actor_token_budget_per_gpu=$(((token_budget + actor_cp - 1) / actor_cp))
ref_token_budget_per_gpu=$(((token_budget + ref_cp - 1) / ref_cp))

if [[ "${START_RAY_CLUSTER}" == "1" ]]; then
    start_ray_cluster
fi

# Use a per-run tool config so concurrent jobs do not rewrite the template.
TOOL_CONFIG_TEMPLATE=${TOOL_CONFIG_TEMPLATE:-${PROJECT_DIR}/examples/sglang_multiturn/config/tool_config/search_tool_config.bcp.yaml}
TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-${PROJECT_DIR}/examples/sglang_multiturn/config/tool_config/search_tool_config.${EXPERIMENT_NAME}.yaml}
cp "${TOOL_CONFIG_TEMPLATE}" "${TOOL_CONFIG_PATH}"
sed -i -E "s#http://[^/]+:[0-9]+/(retrieve|get_doc)#http://${RETRIEVER_HOST}:${RETRIEVER_PORT}/\\1#g" "${TOOL_CONFIG_PATH}"
for worker_ip in ${WORKER_IPS}; do
    rsync -aR -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no" \
        "${PROJECT_DIR}/./verl/tools/search_tool.py" \
        "${PROJECT_DIR}/./verl/tools/open_page_tool.py" \
        "${PROJECT_DIR}/./verl/tools/finish_tool.py" \
        "${PROJECT_DIR}/./verl/tools/utils" \
        "${PROJECT_DIR}/./verl/experimental/agent_loop/agent_loop.py" \
        "${PROJECT_DIR}/./verl/experimental/fully_async_policy/fully_async_trainer.py" \
        "${PROJECT_DIR}/./verl/utils/megatron_utils.py" \
        "${PROJECT_DIR}/./verl/workers/rollout/sglang_rollout/async_sglang_server.py" \
        "${worker_ip}:${PROJECT_DIR}/"
    rsync -aR -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no" \
        "${PROJECT_DIR}/./${TOOL_CONFIG_PATH#${PROJECT_DIR}/}" "${worker_ip}:${PROJECT_DIR}/"
done

RETRIEVER_PID=""
cleanup() {
    if [[ -n "${RETRIEVER_PID}" ]]; then
        kill "${RETRIEVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "${START_RETRIEVER}" == "1" ]]; then
    fuser -k "${RETRIEVER_PORT}/tcp" 2>/dev/null || true
    setsid "${PYTHON_BIN}" "${PROJECT_DIR}/examples/sglang_multiturn/browsecomp_retrieval_server.py" \
        --mode dense \
        --model "${RETRIEVER_MODEL_PATH}" \
        --device cpu \
        --corpus_file "${RETRIEVER_CORPUS_FILE}" \
        --host 0.0.0.0 \
        --port "${RETRIEVER_PORT}" \
        --batch_size 4 \
        --dense_cache "${RETRIEVER_DENSE_CACHE}" \
        > "${LOG_DIR}/browsecomp_retriever.${EXPERIMENT_NAME}.log" 2>&1 &
    RETRIEVER_PID=$!
fi

if [[ "${WAIT_RETRIEVER}" == "1" ]]; then
    for i in $(seq 1 1200); do
        curl --noproxy '*' -sf "http://127.0.0.1:${RETRIEVER_PORT}/health" >/dev/null 2>&1 && break
        if [[ -n "${RETRIEVER_PID}" ]] && ! kill -0 "${RETRIEVER_PID}" 2>/dev/null; then
            cat "${LOG_DIR}/browsecomp_retriever.${EXPERIMENT_NAME}.log" >&2 || true
            exit 1
        fi
        [[ "$i" -eq 1200 ]] && { echo "[retriever] timeout waiting for /health" >&2; exit 1; }
        sleep 1
    done
fi

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.rollout_correction.bypass_mode=${rollout_correction_bypass_mode}
    algorithm.rollout_correction.rollout_is=null
    algorithm.rollout_correction.rollout_rs=null
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_FILE}']"
    data.train_batch_size=0
    data.gen_batch_size=1
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.return_raw_chat=True
    data.return_multi_modal_inputs=False
    +data.apply_chat_template_kwargs.enable_thinking=True
    data.tool_config_path="${TOOL_CONFIG_PATH}"
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6}
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.total_training_steps=${total_training_steps}
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_training_steps}
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BSZ:-1}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_token_budget_per_gpu}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.use_rollout_log_probs=True
    actor_rollout_ref.actor.clip_ratio=0.28
    actor_rollout_ref.actor.clip_ratio_low=0.20
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null
    actor_rollout_ref.actor.megatron.context_parallel_size=${actor_cp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${actor_etp}
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${update_weights_bucket_mb}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BSZ:-1}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${actor_token_budget_per_gpu}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.n=${n_resp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.quantization=${rollout_quantization}
    actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager}
    actor_rollout_ref.rollout.max_model_len=${max_model_len}
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=${max_model_len}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=${sglang_disable_overlap_schedule}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=${sglang_chunked_prefill_size}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_prefill_tokens=${sglang_max_prefill_tokens}
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=${MAX_PARALLEL_CALLS:-5}
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${max_tool_response_length}
    actor_rollout_ref.rollout.multi_turn.format=hermes
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}"
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_LOOP_WORKERS:-16}
    +actor_rollout_ref.rollout.resource_pool_accelerator_type=rollout_node
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BSZ:-1}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ref_token_budget_per_gpu}
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ref_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ref_pp}
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=null
    actor_rollout_ref.ref.megatron.context_parallel_size=${ref_cp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ref_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ref_etp}
    actor_rollout_ref.ref.megatron.param_offload=True
)

REWARD=(
    reward.custom_reward_function.path="${PROJECT_DIR}/verl/utils/reward_score/bc_p_llm_judge.py"
    reward.custom_reward_function.name=compute_score
    reward.reward_model.enable=False
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME:-ksmooth-bcp}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.default_local_dir="${CKPT_DIR}"
    trainer.resume_mode=${RESUME_MODE:-disable}
    trainer.n_gpus_per_node=${TRAIN_GPUS}
    trainer.nnodes=${TRAIN_NNODES}
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}
    trainer.save_freq=${SAVE_FREQ:-10}
    trainer.test_freq=${TEST_FREQ:-5}
    trainer.total_epochs=${TOTAL_EPOCHS:-5}
    +trainer.resource_pool_accelerator_type=train_node
    +trainer.ray_master_port_range="[39000,40000]"
    +trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
    +trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
)

ASYNC=(
    actor_rollout_ref.hybrid_engine=False
    rollout.nnodes=${ROLLOUT_NNODES}
    rollout.n_gpus_per_node=${ROLLOUT_GPUS}
    rollout.n=${n_resp}
    rollout.total_rollout_steps=${total_rollout_steps}
    async_training.require_batches=${require_batches}
    async_training.staleness_threshold=${staleness_threshold}
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step}
    async_training.partial_rollout=${partial_rollout}
    +async_training.drain_before_sync=${drain_before_sync}
    async_training.use_trainer_do_validate=False
)

RAY=(
    +ray_kwargs.ray_init.address="${HEAD_IP}:${RAY_PORT}"
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS="'${CUDA_DEVICE_MAX_CONNECTIONS}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_WORKSPACE_BASE="'${FLASHINFER_WORKSPACE_BASE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.ROLLOUT_MASTER_PORT_RANGE="'${ROLLOUT_MASTER_PORT_RANGE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.SGLANG_SERVER_PORT_RANGE="'${SGLANG_SERVER_PORT_RANGE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.SGLANG_SERVER_PORT_STRIDE="'${SGLANG_SERVER_PORT_STRIDE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_DATAPROTO_SERIALIZATION_METHOD="'${VERL_DATAPROTO_SERIALIZATION_METHOD}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.FULLY_ASYNC_SKIP_OLD_LOGPROB_CPU_SNAPSHOT="'${fully_async_skip_old_logprob_cpu_snapshot}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MEGATRON_CPU_SNAPSHOT_PIN_MEMORY="'${megatron_cpu_snapshot_pin_memory}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.BCP_JUDGE_API_KEY_ENV="'${BCP_JUDGE_API_KEY_ENV}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.BCP_JUDGE_API_BASE="'${BCP_JUDGE_API_BASE}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.BCP_JUDGE_MODEL="'${BCP_JUDGE_MODEL}'"
    +ray_kwargs.ray_init.runtime_env.env_vars.${BCP_JUDGE_API_KEY_ENV}="'${!BCP_JUDGE_API_KEY_ENV}'"
)

if [[ -n "${MCORE_MODEL_PATH}" ]]; then
    ACTOR+=(
        actor_rollout_ref.actor.megatron.dist_checkpointing_path="${MCORE_MODEL_PATH}"
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=True
    )
    REF+=(
        actor_rollout_ref.ref.megatron.dist_checkpointing_path="${MCORE_MODEL_PATH}"
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=True
    )
fi

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path="${PROJECT_DIR}/verl/experimental/fully_async_policy/config" \
    --config-name=fully_async_ppo_megatron_trainer \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${ASYNC[@]}" \
    "${RAY[@]}" \
    "$@"
