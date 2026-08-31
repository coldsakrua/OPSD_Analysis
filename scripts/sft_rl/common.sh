#!/bin/bash
# Shared defaults for scripts/sft_rl GRPO launchers.
# Expected to be sourced from a concrete job script after BASE_DIR is set.

: "${BASE_DIR:?BASE_DIR must be set before sourcing common.sh}"

NUM_GPUS=${NUM_GPUS:-4}
MAX_STEPS=${MAX_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-25}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-12288}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
DAPO_EPSILON=${DAPO_EPSILON:-0.2}
DAPO_EPSILON_HIGH=${DAPO_EPSILON_HIGH:-0.28}
PPO_EPOCHS=${PPO_EPOCHS:-2}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-4096}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}
# Per-step rollout JSONL dump (verl trainer.rollout_data_dir); 0 = dump all.
ROLLOUT_DUMP_N=${ROLLOUT_DUMP_N:-32}

DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/dapo/preprocessed/dapo-math-17k.qwen3.think.maxprompt1024.parquet}
VAL_DATASET_PATH=${VAL_DATASET_PATH:-${DATASET_PATH}}

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-${BASE_DIR}/wandb}
export WANDB_PROJECT=${WANDB_PROJECT:-SFT_RL_GRPO}
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-ERROR}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HYDRA_FULL_ERROR=1
# Do NOT set expandable_segments:True — vLLM CuMemAllocator (sleep/wake) asserts against it.
unset PYTORCH_CUDA_ALLOC_CONF
export VLLM_USE_V1=${VLLM_USE_V1:-0}
# Slurm/ROCm nodes often set both; verl workers refuse that combo.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# This verl build colocates actor+vLLM on all GPUs (hybrid). Split only budgets util.
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
export TRAIN_GPUS=${TRAIN_GPUS:-2}
if [[ $((ROLLOUT_GPUS + TRAIN_GPUS)) -ne "${NUM_GPUS}" ]]; then
  echo "[error] ROLLOUT_GPUS(${ROLLOUT_GPUS})+TRAIN_GPUS(${TRAIN_GPUS}) != NUM_GPUS(${NUM_GPUS})" >&2
  exit 1
fi

MAX_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
