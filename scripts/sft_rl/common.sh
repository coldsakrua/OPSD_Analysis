#!/bin/bash
# Shared defaults for scripts/sft_rl GRPO launchers.
# Expected to be sourced from a concrete job script after BASE_DIR is set.

: "${BASE_DIR:?BASE_DIR must be set before sourcing common.sh}"

NUM_GPUS=${NUM_GPUS:-4}
MAX_STEPS=${MAX_STEPS:-200}
SAVE_STEPS=${SAVE_STEPS:-25}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-20480}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
# Slightly lower LR + 1 PPO epoch to reduce collapse risk with long CoT.
LEARNING_RATE=${LEARNING_RATE:-5e-7}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
DAPO_EPSILON=${DAPO_EPSILON:-0.2}
DAPO_EPSILON_HIGH=${DAPO_EPSILON_HIGH:-0.28}
PPO_EPOCHS=${PPO_EPOCHS:-1}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
# Soft overlong zone: [max_resp - len, max_resp]. With max=20480, len=8192 → 12k–20k.
# penalty = -min(exceed/len, 1) * factor; factor=0.4 → at most -0.4 at the hard max.
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-8192}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-0.4}
OVERLONG_PENALTY_ENABLE=${OVERLONG_PENALTY_ENABLE:-true}
# GRPO-style KL against frozen ref (needs RefPolicy worker).
USE_KL_LOSS=${USE_KL_LOSS:-true}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}
# Offload ref params to CPU between forwards (saves GPU mem with 20k + hybrid vLLM).
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-true}
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

# Cluster soft nproc≈200; Ray/vLLM spawn many procs under node packing → EAGAIN / dashboard EOF.
ulimit -u 8192 2>/dev/null || true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
# Keep all GPUs visible; verl Worker.set_device(RAY_LOCAL_RANK). Avoids NCCL
# Duplicate GPU when CUDA inits before Ray remaps CUDA_VISIBLE_DEVICES.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}
# Keep Ray within Slurm CPU allocation (default 7 CPU/GPU on GPUA800).
RAY_NUM_CPUS=${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-$((NUM_GPUS * 7))}}
export RAY_NUM_CPUS
echo "[grpo] nproc_limit=$(ulimit -u) OMP=${OMP_NUM_THREADS} ray_num_cpus=${RAY_NUM_CPUS} NOSET_CVD=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES}"

# This verl build colocates actor+vLLM on all GPUs (hybrid). Split only budgets util.
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
export TRAIN_GPUS=${TRAIN_GPUS:-2}
if [[ $((ROLLOUT_GPUS + TRAIN_GPUS)) -ne "${NUM_GPUS}" ]]; then
  echo "[error] ROLLOUT_GPUS(${ROLLOUT_GPUS})+TRAIN_GPUS(${TRAIN_GPUS}) != NUM_GPUS(${NUM_GPUS})" >&2
  exit 1
fi

MAX_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
