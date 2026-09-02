#!/bin/bash
# Internal launcher: expects MODEL_PATH / RUN_NAME / MODEL_TAG / ROLLOUT_GPUS / TRAIN_GPUS
# and batch-size knobs already set by the caller.
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
# shellcheck source=/dev/null
source "${BASE_DIR}/scripts/sft_rl/common.sh"

: "${MODEL_PATH:?MODEL_PATH required}"
: "${RUN_NAME:?RUN_NAME required}"
: "${MODEL_TAG:?MODEL_TAG required}"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-}
VLLM_TP_SIZE=${VLLM_TP_SIZE:-1}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-}

OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/${MODEL_TAG}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/${JOB_TAG}}
WANDB_NAME=${WANDB_NAME:-${RUN_NAME}_${JOB_TAG}}
WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-${RUN_NAME}}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_HOME}" "${BASE_DIR}/log/train/sft_rl"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing dataset: ${DATASET_PATH}" >&2
  echo "[error] run: sbatch scripts/sft_rl/preprocess_dapo_grpo.sh" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[error] missing model: ${MODEL_PATH}" >&2
  exit 1
fi

UTIL_ARGS=()
if [[ -n "${VLLM_GPU_MEM_UTIL}" ]]; then
  UTIL_ARGS+=(actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEM_UTIL}")
fi
CHAT_ARGS=()
if [[ -n "${CHAT_TEMPLATE_PATH}" ]]; then
  CHAT_ARGS+=(++opsd.chat_template_path="${CHAT_TEMPLATE_PATH}")
fi

echo "[launch] run=${WANDB_NAME}"
echo "[launch] model=${MODEL_PATH}"
echo "[launch] dataset=${DATASET_PATH}"
echo "[launch] output=${OUTPUT_DIR}"
echo "[launch] gpus=${NUM_GPUS} intended_split=rollout${ROLLOUT_GPUS}+train${TRAIN_GPUS} (hybrid colocated)"
echo "[launch] steps=${MAX_STEPS} save=${SAVE_STEPS} n=${NUM_GENERATIONS} resp=${MAX_RESPONSE_LENGTH}"
echo "[launch] train_bs=${TRAIN_BATCH_SIZE} mini=${PPO_MINI_BATCH_SIZE} epochs=${PPO_EPOCHS} lr=${LEARNING_RATE}"
echo "[launch] vllm_util=${VLLM_GPU_MEM_UTIL:-auto} free_cache_engine=true overlong_penalty=${OVERLONG_PENALTY_ENABLE} len=${OVERLONG_BUFFER_LEN} factor=${OVERLONG_PENALTY_FACTOR}"
echo "[launch] kl_loss=${USE_KL_LOSS} kl_coef=${KL_LOSS_COEF} kl_type=${KL_LOSS_TYPE} ref_param_offload=${REF_PARAM_OFFLOAD}"
echo "[launch] reward=src/reward_math_dapo_boxed.py (boxed-first) + stop_token_ids auto from tokenizer"
echo "[launch] rollout_dump=${OUTPUT_DIR}/rollouts max_per_step=${ROLLOUT_DUMP_N:-32}"

python -u "${BASE_DIR}/src/train_grpo_dapo.py" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  algorithm.norm_adv_by_std_in_grpo=true \
  data.train_files="${DATASET_PATH}" \
  data.val_files="${VAL_DATASET_PATH}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  data.return_raw_chat=true \
  ++data.apply_chat_template_kwargs.enable_thinking=true \
  actor_rollout_ref.hybrid_engine=true \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}" \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}" \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.clip_ratio="${DAPO_EPSILON}" \
  ++actor_rollout_ref.actor.clip_ratio_low="${DAPO_EPSILON}" \
  ++actor_rollout_ref.actor.clip_ratio_high="${DAPO_EPSILON_HIGH}" \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}" \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.disable_log_stats=true \
  actor_rollout_ref.rollout.n="${NUM_GENERATIONS}" \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
  actor_rollout_ref.rollout.top_k="${TOP_K}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${VLLM_TP_SIZE}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_LENGTH}" \
  reward_model.enable=false \
  reward_model.reward_manager=dapo \
  ++reward_model.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}" \
  ++reward_model.reward_kwargs.overlong_buffer_cfg.enable="${OVERLONG_PENALTY_ENABLE}" \
  ++reward_model.reward_kwargs.overlong_buffer_cfg.len="${OVERLONG_BUFFER_LEN}" \
  ++reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor="${OVERLONG_PENALTY_FACTOR}" \
  ++reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
  ++custom_reward_function.path="${BASE_DIR}/src/reward_math_dapo_boxed.py" \
  ++custom_reward_function.name=compute_score \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.total_training_steps="${MAX_STEPS}" \
  trainer.save_freq="${SAVE_STEPS}" \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.critic_warmup=0 \
  trainer.logger='[console,wandb]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${WANDB_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.resume_mode=disable \
  ++trainer.rollout_data_dir="${OUTPUT_DIR}/rollouts" \
  ++opsd.rollout_dump_n="${ROLLOUT_DUMP_N:-32}" \
  ++opsd.rollout_gpus="${ROLLOUT_GPUS}" \
  ++opsd.train_gpus="${TRAIN_GPUS}" \
  ray_init.num_cpus="${RAY_NUM_CPUS}" \
  hydra.run.dir="${OUTPUT_DIR}/hydra" \
  hydra.job.chdir=false \
  "${UTIL_ARGS[@]}" \
  "${CHAT_ARGS[@]}"
