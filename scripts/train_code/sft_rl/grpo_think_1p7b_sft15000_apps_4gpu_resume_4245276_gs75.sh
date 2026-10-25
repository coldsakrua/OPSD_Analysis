#!/bin/bash
#SBATCH --job-name=grpo_1p7b_apps_r75
#SBATCH --output=log/train_code/sft_rl/grpo_think_1p7b_ocr_sft15000_apps_4gpu_resume_gs75.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=240:00:00
set -euo pipefail

# Resume grpo_think_1p7b_ocr_sft15000_apps_4gpu (jid 4245276) from global_step_75
# after CUDA illegal-memory failure ~step 81. Continues to MAX_STEPS=100 in the
# same output dir (new checkpoints land under .../4245276/).

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
export NUM_GPUS=${NUM_GPUS:-4}
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-3}
export TRAIN_GPUS=${TRAIN_GPUS:-1}

MODEL_PATH=${MODEL_PATH:-${BASE_DIR}/outputs/qwen3_1.7b_base/sft_think_4gpu_12k_ocr/3925961/checkpoint-15000}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b}
MODEL_TAG=${MODEL_TAG:-qwen3_1.7b_base}
RUN_NAME=${RUN_NAME:-grpo_think_1p7b_ocr_sft15000_apps_4gpu}

# Keep writing into the failed run's directory.
RESUME_ROOT=${RESUME_ROOT:-${BASE_DIR}/outputs/qwen3_1.7b_base/grpo_think_1p7b_ocr_sft15000_apps_4gpu/4245276}
OUTPUT_DIR=${OUTPUT_DIR:-${RESUME_ROOT}}
RESUME_MODE=${RESUME_MODE:-resume_path}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-${RESUME_ROOT}/global_step_75}

if [[ ! -d "${RESUME_FROM_PATH}/actor" ]]; then
  echo "[error] missing resume actor: ${RESUME_FROM_PATH}/actor" >&2
  exit 1
fi
if [[ ! -f "${RESUME_ROOT}/latest_checkpointed_iteration.txt" ]]; then
  echo "[error] missing latest_checkpointed_iteration.txt under ${RESUME_ROOT}" >&2
  exit 1
fi

NUM_GENERATIONS=${NUM_GENERATIONS:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-24576}
MAX_STEPS=${MAX_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-25}
OVERLONG_PENALTY_ENABLE=${OVERLONG_PENALTY_ENABLE:-false}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-96}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-48}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-26624}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-106496}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.90}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-true}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-true}

APPS_REWARD_WORKERS=${APPS_REWARD_WORKERS:-24}

DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/apps/preprocessed/apps.traintest.qwen3.think.maxprompt${MAX_PROMPT_LENGTH}.maxtests10.parquet}
VAL_DATASET_PATH=${VAL_DATASET_PATH:-${DATASET_PATH}}

export WANDB_PROJECT=${WANDB_PROJECT:-SFT_RL_GRPO_APPS}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_1p7b_ocr_sft15000_grpo_apps}
# Distinct wandb name; checkpoints still go to OUTPUT_DIR=4245276.
export WANDB_NAME=${WANDB_NAME:-${RUN_NAME}_resume_gs75_${SLURM_JOB_ID:-manual}}

export BASE_DIR MODEL_PATH CHAT_TEMPLATE_PATH MODEL_TAG RUN_NAME
export OUTPUT_DIR RESUME_MODE RESUME_FROM_PATH
export NUM_GENERATIONS MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH
export MAX_STEPS SAVE_STEPS OVERLONG_PENALTY_ENABLE
export TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU PPO_MAX_TOKEN_LEN_PER_GPU MAX_NUM_BATCHED_TOKENS
export VLLM_GPU_MEM_UTIL ACTOR_PARAM_OFFLOAD ACTOR_OPTIMIZER_OFFLOAD
export APPS_REWARD_WORKERS DATASET_PATH VAL_DATASET_PATH WANDB_NAME

echo "[resume] from=${RESUME_FROM_PATH}"
echo "[resume] output_dir=${OUTPUT_DIR} (same as failed jid 4245276)"
echo "[resume] target_steps=${MAX_STEPS}"

exec bash "${BASE_DIR}/scripts/train_code/sft_rl/_run_grpo_apps.sh"
