#!/bin/bash
#SBATCH --job-name=grpo_4b_omr
#SBATCH --output=log/train/rl/qwen3_4b_think/grpo_think_4b_omr_int_4gpu_bs64_24k.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=120:00:00
set -euo pipefail

# Qwen3-4B think-mode GRPO. Optimizer schedule matches 1.7B 4gpu@24k:
#   train_bs=64 × max_steps=100 → 6400 prompts
#   mini=64 → one optimizer.step per trainer step
# 4B@4gpu needs more packing-limited grad accum than 1.7B:
#   ppo_max_tok 36864 → 25600 (= max_model_len, 1 full seq / GPU update)
#   logprob micro 2 → 1
# Rollout packing (full-seq): batched 25600 → 102400 (4 × 25600). Qwen3-4B GQA KV
#   ≈ 3.6GB/seq; 4 seqs ≈ 14GB + 8GB weights, well under util=0.90.
# Hybrid colocated: rollout2+train2. Actor offload required (4B@24k OOM without it).

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
export NUM_GPUS=${NUM_GPUS:-4}
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
export TRAIN_GPUS=${TRAIN_GPUS:-2}

MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-${MODEL_PATH}}
MODEL_TAG=${MODEL_TAG:-qwen3_4b}
RUN_NAME=${RUN_NAME:-grpo_think_4b_omr_int_4gpu_bs64_24k_6400}

DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openmathreasoning/preprocessed/omr.integer_answer.think.le12k.grpo.parquet}
VAL_DATASET_PATH=${VAL_DATASET_PATH:-${DATASET_PATH}}

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
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
# max_model_len ≈ 1024+24576=25600. Floor token budget → 1 full seq / GPU → more accum.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-25600}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-102400}
# Rollout/update sequential (vLLM sleep + actor offload). Keep high util.
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.90}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-true}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-true}

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_4b_grpo_omr_int_4gpu_bs64_24k_6400}
export WANDB_PROJECT=${WANDB_PROJECT:-SFT_RL_GRPO}

export BASE_DIR MODEL_PATH CHAT_TEMPLATE_PATH MODEL_TAG RUN_NAME
export DATASET_PATH VAL_DATASET_PATH
export TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE MAX_STEPS SAVE_STEPS NUM_GENERATIONS
export MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH LEARNING_RATE
export OVERLONG_PENALTY_ENABLE OVERLONG_BUFFER_LEN OVERLONG_PENALTY_FACTOR
export PPO_MICRO_BATCH_SIZE_PER_GPU LOG_PROB_MICRO_BATCH_SIZE_PER_GPU
export PPO_MAX_TOKEN_LEN_PER_GPU MAX_NUM_BATCHED_TOKENS VLLM_GPU_MEM_UTIL
export ACTOR_PARAM_OFFLOAD ACTOR_OPTIMIZER_OFFLOAD

mkdir -p "${BASE_DIR}/log/train/rl/qwen3_4b_think"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing GRPO dataset: ${DATASET_PATH}" >&2
  echo "[error] after SFT le12k filter, run: sbatch scripts/data/run_preprocess_omr_integer_grpo.sh" >&2
  exit 1
fi

TARGET_PROMPTS=$((TRAIN_BATCH_SIZE * MAX_STEPS))
echo "[launch] prompts_seen=${TARGET_PROMPTS} (train_bs=${TRAIN_BATCH_SIZE} × steps=${MAX_STEPS}; match OPSD / 1.7B 6400)"
echo "[launch] update schedule: train_bs=${TRAIN_BATCH_SIZE} mini=${PPO_MINI_BATCH_SIZE} → 1 optimizer.step / trainer step"
echo "[launch] grad accum: ppo_max_tok=${PPO_MAX_TOKEN_LEN_PER_GPU} logprob_micro=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} (more accum vs 1.7B 36864/2)"
echo "[launch] n=${NUM_GENERATIONS} resp=${MAX_RESPONSE_LENGTH} lr=${LEARNING_RATE} util=${VLLM_GPU_MEM_UTIL} batched=${MAX_NUM_BATCHED_TOKENS} (=4×25600)"
echo "[launch] actor_offload param=${ACTOR_PARAM_OFFLOAD} optim=${ACTOR_OPTIMIZER_OFFLOAD}"

exec bash "${BASE_DIR}/scripts/sft_rl/_run_grpo.sh"
