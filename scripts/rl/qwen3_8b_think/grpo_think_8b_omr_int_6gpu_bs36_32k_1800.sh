#!/bin/bash
#SBATCH --job-name=grpo_8b_omr_6g32k
#SBATCH --output=log/train/rl/qwen3_8b_think/grpo_think_8b_omr_int_6gpu_bs36_32k.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --gres=gpu:6
#SBATCH --mem=600G
#SBATCH --time=120:00:00
set -euo pipefail

# Qwen3-8B think GRPO @6gpu, resp=32k (try to cut 24k clip_ratio).
#
# Hybrid colocated: all 6 GPUs generate, then all 6 FSDP-update (not a physical
# rollout/train pool split). ROLLOUT_GPUS+TRAIN_GPUS only labels util budgeting.
#
# vs 4gpu@24k:
#   max_model_len 25600 → 33792 (prompt1024+resp32768)
#   update floor stays 1 full seq/GPU → ppo_max_tok=33792
#   generate: start with 1× full seq batched (safer); raise to 2× if stable
#   util=0.90 (same as 8B@4gpu 24k; drop to 0.85 only if wake OOM)
#   train_bs=36: train_bs*n(=8) divisible by 6; 36×50 = 1800 prompts
# Keep param/optim/activation offload — 24k already needed act offload on update.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
export NUM_GPUS=${NUM_GPUS:-6}
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-3}
export TRAIN_GPUS=${TRAIN_GPUS:-3}

MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-8b}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-${MODEL_PATH}}
MODEL_TAG=${MODEL_TAG:-qwen3_8b}
RUN_NAME=${RUN_NAME:-grpo_think_8b_omr_int_6gpu_bs36_32k_1800}

DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openmathreasoning/preprocessed/omr.integer_answer.think.le12k.grpo.parquet}
VAL_DATASET_PATH=${VAL_DATASET_PATH:-${DATASET_PATH}}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-36}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-36}
MAX_STEPS=${MAX_STEPS:-50}
SAVE_STEPS=${SAVE_STEPS:-25}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
OVERLONG_PENALTY_ENABLE=${OVERLONG_PENALTY_ENABLE:-false}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-4096}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-0.4}

PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
# Update floor = max_model_len = 1 full 32k seq / GPU.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-33792}
# Generate: 1× full 32k first (KV≈4.6GB/seq). If step1 stable, try 67584 (=2×).
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-33792}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.90}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-true}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-true}
ENABLE_ACTIVATION_OFFLOAD=${ENABLE_ACTIVATION_OFFLOAD:-true}

export OPSD_NCCL_TIMEOUT_SEC=${OPSD_NCCL_TIMEOUT_SEC:-14400}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-${OPSD_NCCL_TIMEOUT_SEC}}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-${OPSD_NCCL_TIMEOUT_SEC}}

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_8b_grpo_omr_int_6gpu_bs36_32k_1800}
export WANDB_PROJECT=${WANDB_PROJECT:-SFT_RL_GRPO}

export BASE_DIR MODEL_PATH CHAT_TEMPLATE_PATH MODEL_TAG RUN_NAME
export DATASET_PATH VAL_DATASET_PATH
export TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE MAX_STEPS SAVE_STEPS NUM_GENERATIONS
export MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH LEARNING_RATE
export OVERLONG_PENALTY_ENABLE OVERLONG_BUFFER_LEN OVERLONG_PENALTY_FACTOR
export PPO_MICRO_BATCH_SIZE_PER_GPU LOG_PROB_MICRO_BATCH_SIZE_PER_GPU
export PPO_MAX_TOKEN_LEN_PER_GPU MAX_NUM_BATCHED_TOKENS VLLM_GPU_MEM_UTIL
export ACTOR_PARAM_OFFLOAD ACTOR_OPTIMIZER_OFFLOAD ENABLE_ACTIVATION_OFFLOAD
export OPSD_NCCL_TIMEOUT_SEC TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC NCCL_TIMEOUT

mkdir -p "${BASE_DIR}/log/train/rl/qwen3_8b_think"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing GRPO dataset: ${DATASET_PATH}" >&2
  echo "[error] after SFT le12k filter, run: sbatch scripts/data/run_preprocess_omr_integer_grpo.sh" >&2
  exit 1
fi

# train_bs * n must be divisible by n_gpus.
if (( (TRAIN_BATCH_SIZE * NUM_GENERATIONS) % NUM_GPUS != 0 )); then
  echo "[error] train_bs(${TRAIN_BATCH_SIZE})*n(${NUM_GENERATIONS}) not divisible by n_gpus(${NUM_GPUS})" >&2
  exit 1
fi
if (( TRAIN_BATCH_SIZE % 3 != 0 )); then
  echo "[error] for 6gpu+n=8 need train_bs % 3 == 0 (got ${TRAIN_BATCH_SIZE})" >&2
  exit 1
fi

TARGET_PROMPTS=$((TRAIN_BATCH_SIZE * MAX_STEPS))
echo "[launch] prompts_seen=${TARGET_PROMPTS} (train_bs=${TRAIN_BATCH_SIZE} × steps=${MAX_STEPS})"
echo "[launch] update: mini=${PPO_MINI_BATCH_SIZE} → 1 optimizer.step / trainer step; ppo_max_tok=${PPO_MAX_TOKEN_LEN_PER_GPU} (=1×full-32k/GPU)"
echo "[launch] n=${NUM_GENERATIONS} resp=${MAX_RESPONSE_LENGTH} util=${VLLM_GPU_MEM_UTIL} batched=${MAX_NUM_BATCHED_TOKENS}"
echo "[launch] offload param=${ACTOR_PARAM_OFFLOAD} optim=${ACTOR_OPTIMIZER_OFFLOAD} activation=${ENABLE_ACTIVATION_OFFLOAD}"
echo "[launch] if vLLM wake OOM: util=0.85; if update OOM: keep offloads; if gen slow but update OK: batched=67584"

exec bash "${BASE_DIR}/scripts/sft_rl/_run_grpo.sh"
