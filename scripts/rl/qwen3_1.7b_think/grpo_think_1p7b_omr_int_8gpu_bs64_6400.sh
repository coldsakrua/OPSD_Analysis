#!/bin/bash
#SBATCH --job-name=grpo_1p7b_omr8
#SBATCH --output=log/train/rl/qwen3_1.7b_think/grpo_think_1p7b_omr_int_8gpu_bs64.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:8
#SBATCH --mem=800G
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n13
set -euo pipefail

# 8-GPU expansion of grpo_think_1p7b_omr_int_4gpu_bs64_6400 (job 3567994 @16k).
# Keep optimizer schedule identical:
#   train_bs=64 × max_steps=100 → 6400 prompts
#   mini=64 → one optimizer.step per trainer step
# Extra GPUs buy longer rollout: max_response 16k → 24k (max_model_len≈25600).
# Hybrid colocated: intended split rollout3+train5 (same as other 1.7B/4B@8gpu).
# Actor param/optim offload kept — 4gpu@16k OOM'd on vLLM wake without it.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
export NUM_GPUS=${NUM_GPUS:-8}
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-3}
export TRAIN_GPUS=${TRAIN_GPUS:-5}

MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-${MODEL_PATH}}
MODEL_TAG=${MODEL_TAG:-qwen3_1.7b}
RUN_NAME=${RUN_NAME:-grpo_think_1p7b_omr_int_8gpu_bs64_24k_6400}

DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openmathreasoning/preprocessed/omr.integer_answer.think.le12k.grpo.parquet}
VAL_DATASET_PATH=${VAL_DATASET_PATH:-${DATASET_PATH}}

# Same update schedule as 4gpu_bs64; only response length + GPU count change.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MAX_STEPS=${MAX_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-25}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-24576}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
OVERLONG_PENALTY_ENABLE=${OVERLONG_PENALTY_ENABLE:-false}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-4096}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-0.4}

PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
# max_model_len ≈ 1024+24576=25600; leave packing headroom for dynamic bsz.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-36864}
# 8gpu@16k used batched=49152 (65536 OOM'd); keep that for 24k.
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-49152}
# Rollout/update sequential (vLLM sleep); util can stay high with actor offload.
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.90}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-true}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-true}

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_1p7b_grpo_omr_int_8gpu_bs64_24k_6400}
export WANDB_PROJECT=${WANDB_PROJECT:-SFT_RL_GRPO}

export BASE_DIR MODEL_PATH CHAT_TEMPLATE_PATH MODEL_TAG RUN_NAME
export DATASET_PATH VAL_DATASET_PATH
export TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE MAX_STEPS SAVE_STEPS NUM_GENERATIONS
export MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH LEARNING_RATE
export OVERLONG_PENALTY_ENABLE OVERLONG_BUFFER_LEN OVERLONG_PENALTY_FACTOR
export PPO_MICRO_BATCH_SIZE_PER_GPU LOG_PROB_MICRO_BATCH_SIZE_PER_GPU
export PPO_MAX_TOKEN_LEN_PER_GPU MAX_NUM_BATCHED_TOKENS VLLM_GPU_MEM_UTIL
export ACTOR_PARAM_OFFLOAD ACTOR_OPTIMIZER_OFFLOAD

mkdir -p "${BASE_DIR}/log/train/rl/qwen3_1.7b_think"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing GRPO dataset: ${DATASET_PATH}" >&2
  echo "[error] after SFT le12k filter, run: sbatch scripts/data/run_preprocess_omr_integer_grpo.sh" >&2
  exit 1
fi

TARGET_PROMPTS=$((TRAIN_BATCH_SIZE * MAX_STEPS))
echo "[launch] prompts_seen=${TARGET_PROMPTS} (train_bs=${TRAIN_BATCH_SIZE} × steps=${MAX_STEPS}; match OPSD 6400)"
echo "[launch] update schedule: train_bs=${TRAIN_BATCH_SIZE} mini=${PPO_MINI_BATCH_SIZE} → 1 optimizer.step / trainer step (grad accum over micros)"
echo "[launch] n=${NUM_GENERATIONS} resp=${MAX_RESPONSE_LENGTH} lr=${LEARNING_RATE}"
echo "[launch] util=${VLLM_GPU_MEM_UTIL} ppo_max_tok=${PPO_MAX_TOKEN_LEN_PER_GPU} batched=${MAX_NUM_BATCHED_TOKENS}"
echo "[launch] actor_offload param=${ACTOR_PARAM_OFFLOAD} optim=${ACTOR_OPTIMIZER_OFFLOAD}"

exec bash "${BASE_DIR}/scripts/sft_rl/_run_grpo.sh"
