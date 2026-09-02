#!/bin/bash
#SBATCH --job-name=grpo_4b_s15k
#SBATCH --output=log/train/sft_rl/grpo_think_4b_sft15000_8gpu.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:8
#SBATCH --mem=800G
#SBATCH --time=160:00:00
set -euo pipefail

# Qwen3-4B SFT@15000 → GRPO on 8 GPUs.
# 20k response + ref KL + overlong from 12k. Conservative util for hybrid+ref.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
export NUM_GPUS=${NUM_GPUS:-8}
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-3}
export TRAIN_GPUS=${TRAIN_GPUS:-5}

MODEL_PATH=${MODEL_PATH:-${BASE_DIR}/outputs/qwen3_4b_base/checkpoint-15000}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
MODEL_TAG=${MODEL_TAG:-qwen3_4b_base}
RUN_NAME=${RUN_NAME:-grpo_think_4b_sft15000_8gpu}

# n=8 → any train_bs works on 8 GPUs. Keep small for 20k + RefPolicy.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
# max_model_len ≈ 1024+20480=21504; leave headroom for packing.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
# Keep below 20k*2: util=0.90 OOMed on step2 kv wake with RefPolicy.
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-24576}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.75}

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_4b_sft15000_grpo_dapo}

export BASE_DIR MODEL_PATH CHAT_TEMPLATE_PATH MODEL_TAG RUN_NAME
export TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU PPO_MAX_TOKEN_LEN_PER_GPU MAX_NUM_BATCHED_TOKENS
export VLLM_GPU_MEM_UTIL

exec bash "${BASE_DIR}/scripts/sft_rl/_run_grpo.sh"
