#!/bin/bash
#SBATCH --job-name=grpo_olmo7bt
#SBATCH --output=log/train/rl/olmo3_7b_think/grpo_think_olmo7bt_omr_int_6gpu_bs64_24k.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --gres=gpu:6
#SBATCH --mem=600G
#SBATCH --time=120:00:00
set -euo pipefail

# Olmo-3-7B-Think GRPO, cloned from
#   scripts/rl/qwen3_1.7b_think/grpo_think_1p7b_omr_int_4gpu_bs64_24k_6400.sh
# Single node 6-GPU (not 2×4). Optimizer schedule unchanged vs 1.7B / OPSD:
#   train_bs=64 × max_steps=100 → 6400 prompts
#   mini=64 → one optimizer.step per trainer step
# Hybrid colocated: intended split rollout2+train4 (same as other 6gpu jobs).
# Actor offload required (7B@24k OOM on vLLM wake without it).
#
# Memory (A800 80GB, util=0.90, assume every rollout is max_model_len=25600):
#   Olmo-3-7B full MHA KV ≈ 0.5MB/token → 12.8GB/seq; weights ≈14GB
#   vLLM budget ≈72GB → ~4 full seqs of KV; leave headroom for prefill
#   Actor backward is tighter than decode → pack 2 seqs, not 3.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
export NUM_GPUS=${NUM_GPUS:-6}
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
export TRAIN_GPUS=${TRAIN_GPUS:-4}

MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/olmo-3-7b-think}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-${MODEL_PATH}}
MODEL_TAG=${MODEL_TAG:-olmo3_7b_think}
RUN_NAME=${RUN_NAME:-grpo_think_olmo7bt_omr_int_6gpu_bs64_24k_6400}

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
# Official Olmo-3-Think sampling: temp=0.6 top_p=0.95, no top-k.
TOP_K=${TOP_K:--1}

PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
# max_model_len ≈ 1024+24576=25600. Full-seq packing (not 25600=1 seq):
#   update/logprob: 2 × 25600 = 51200  (backward + flash-attn checkpoint)
#   vLLM decode:    3 × 25600 = 76800  (KV≈38GB + 14GB weights, ~52/72GB)
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-51200}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-76800}
# Rollout/update are sequential (vLLM sleeps during update + actor offload).
# GPU is otherwise free in the rollout phase — keep high util like 1.7B.
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.90}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-true}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-true}

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-olmo3_7b_think_grpo_omr_int_6gpu_bs64_24k_6400}
export WANDB_PROJECT=${WANDB_PROJECT:-SFT_RL_GRPO}

export BASE_DIR MODEL_PATH CHAT_TEMPLATE_PATH MODEL_TAG RUN_NAME
export DATASET_PATH VAL_DATASET_PATH
export TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE MAX_STEPS SAVE_STEPS NUM_GENERATIONS
export MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH LEARNING_RATE TOP_K
export OVERLONG_PENALTY_ENABLE OVERLONG_BUFFER_LEN OVERLONG_PENALTY_FACTOR
export PPO_MICRO_BATCH_SIZE_PER_GPU LOG_PROB_MICRO_BATCH_SIZE_PER_GPU
export PPO_MAX_TOKEN_LEN_PER_GPU MAX_NUM_BATCHED_TOKENS VLLM_GPU_MEM_UTIL
export ACTOR_PARAM_OFFLOAD ACTOR_OPTIMIZER_OFFLOAD

mkdir -p "${BASE_DIR}/log/train/rl/olmo3_7b_think"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing GRPO dataset: ${DATASET_PATH}" >&2
  echo "[error] after SFT le12k filter, run: sbatch scripts/data/run_preprocess_omr_integer_grpo.sh" >&2
  exit 1
fi

TARGET_PROMPTS=$((TRAIN_BATCH_SIZE * MAX_STEPS))
echo "[launch] prompts_seen=${TARGET_PROMPTS} (train_bs=${TRAIN_BATCH_SIZE} × steps=${MAX_STEPS}; match OPSD 6400)"
echo "[launch] update schedule: train_bs=${TRAIN_BATCH_SIZE} mini=${PPO_MINI_BATCH_SIZE} → 1 optimizer.step / trainer step"
echo "[launch] packing (full-seq): ppo_max_tok=${PPO_MAX_TOKEN_LEN_PER_GPU} (=2×25600) batched=${MAX_NUM_BATCHED_TOKENS} (=3×25600) logprob_micro=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
echo "[launch] n=${NUM_GENERATIONS} resp=${MAX_RESPONSE_LENGTH} lr=${LEARNING_RATE} top_k=${TOP_K} util=${VLLM_GPU_MEM_UTIL}"
echo "[launch] actor_offload param=${ACTOR_PARAM_OFFLOAD} optim=${ACTOR_OPTIMIZER_OFFLOAD}"

exec bash "${BASE_DIR}/scripts/sft_rl/_run_grpo.sh"
