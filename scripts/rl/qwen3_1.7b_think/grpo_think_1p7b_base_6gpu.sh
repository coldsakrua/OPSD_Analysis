#!/bin/bash
#SBATCH --job-name=grpo_1p7b_base
#SBATCH --output=log/train/rl/qwen3_1.7b_think/grpo_think_1p7b_base_6gpu.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --gres=gpu:6
#SBATCH --mem=600G
#SBATCH --time=240:00:00
set -euo pipefail

# Qwen3-1.7B-Base full-param GRPO (think mode). Same pipeline as sft_rl 4B:
# scripts/sft_rl/{common.sh,_run_grpo.sh} → src/train_grpo_dapo.py
# Rollout length / sampling / KL / overlong match 4B (common.sh: resp=20480, n=8).
# 6-GPU hybrid colocated; intended split 2+4. Batch/token knobs sized for 1.7B VRAM.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
export NUM_GPUS=${NUM_GPUS:-6}
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
export TRAIN_GPUS=${TRAIN_GPUS:-4}

MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b-base}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b}
MODEL_TAG=${MODEL_TAG:-qwen3_1.7b_base}
RUN_NAME=${RUN_NAME:-grpo_think_1p7b_base_6gpu}

# ~2–2.5× 4B@6gpu (train_bs=24): 1.7B weights leave headroom for 20k CoT + RefPolicy.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
# max_model_len ≈ 1024+20480=21504; pack denser than 4B's 24576.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-49152}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-65536}
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.90}
# Community/DAPO default; common.sh default is 5e-7.
LEARNING_RATE=${LEARNING_RATE:-1e-6}

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_1p7b_base_grpo_dapo}

export BASE_DIR MODEL_PATH CHAT_TEMPLATE_PATH MODEL_TAG RUN_NAME
export TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU PPO_MAX_TOKEN_LEN_PER_GPU MAX_NUM_BATCHED_TOKENS
export VLLM_GPU_MEM_UTIL LEARNING_RATE

exec bash "${BASE_DIR}/scripts/sft_rl/_run_grpo.sh"
