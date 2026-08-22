#!/bin/bash
#SBATCH --job-name=sft_1p7b_4g_12k
#SBATCH --output=log/train/sft/1.7b_base/sft_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=168:00:00
set -euo pipefail

# 4-GPU formal (scaled from 1-GPU smoke):
#   micro=2, gas=8 → gbs=64 (1GPU smoke was micro=1/gbs=8; 1GPU@8k used ~59/80GB)
#   ZeRO-2 no CPU offload (1.7B 不需 ZeRO-3；省通信、ckpt 更快)
#   max_steps=15000, save_steps=500
#   dataset: offline-tokenized le12k.all.tok
# Resume (same output dir, inherit model/optim/data position):
#   RESUME_FROM_CHECKPOINT=latest OUTPUT_DIR=.../3299217 sbatch scripts/sft/qwen3_1.7b_base/sft_think_4gpu_12k.sh
#   or use scripts/sft/qwen3_1.7b_base/sft_think_4gpu_12k_resume.sh

NUM_GPUS=${NUM_GPUS:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
MAX_LENGTH=${MAX_LENGTH:-12288}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH:-64}
MAX_STEPS=${MAX_STEPS:-15000}
SAVE_STEPS=${SAVE_STEPS:-500}
# Keep every checkpoint (do not set save_total_limit).
RUN_NAME=${RUN_NAME:-sft_think_4gpu_12k}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b-base}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openmathreasoning/preprocessed/omr.cot.think.le12k.all.tok}
MODEL_TAG=${MODEL_TAG:-qwen3_1.7b_base}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/${MODEL_TAG}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/${JOB_TAG}}
RUN_NAME_WITH_JOB=${RUN_NAME}_${JOB_TAG}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export WANDB_MODE=offline
export WANDB_PROJECT=${WANDB_PROJECT:-SFT_OpenMath}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_1p7b_base_sft_fullparam_12k}
export WANDB_DIR=${WANDB_DIR:-${BASE_DIR}/wandb}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_HOME}" "${BASE_DIR}/log/train/sft/1.7b_base"

if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "[error] missing pretokenized dataset dir: ${DATASET_PATH}" >&2
  echo "[error] run: sbatch scripts/sft/qwen3_1.7b_base/tokenize_le12k.sh" >&2
  exit 1
fi

MASTER_PORT=${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}
GLOBAL_BATCH=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))
TOTAL_SAMPLES=$((GLOBAL_BATCH * MAX_STEPS))
if [[ "${GLOBAL_BATCH}" -ne "${TARGET_GLOBAL_BATCH}" ]]; then
  echo "[error] global_batch=${GLOBAL_BATCH} != TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH}" >&2
  exit 1
fi

echo "[launch] run=${RUN_NAME_WITH_JOB} gpus=${NUM_GPUS} global_batch=${GLOBAL_BATCH} total_samples=${TOTAL_SAMPLES}"
echo "[launch] lr=${LEARNING_RATE} max_length=${MAX_LENGTH} max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS}"
echo "[launch] model=${MODEL_PATH}"
echo "[launch] dataset=${DATASET_PATH}"
echo "[launch] output=${OUTPUT_DIR} master_port=${MASTER_PORT}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "[launch] resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}"
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

accelerate launch \
  --config_file "${BASE_DIR}/configs/accelerate_zero2_no_offload.yaml" \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MASTER_PORT}" \
  "${BASE_DIR}/src/train_sft.py" \
  --model-path "${MODEL_PATH}" \
  --chat-template-path "${CHAT_TEMPLATE_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME_WITH_JOB}" \
  --enable-thinking \
  --max-steps "${MAX_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --max-length "${MAX_LENGTH}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --deepspeed "${BASE_DIR}/configs/deepspeed_zero2_no_offload.json" \
  "${RESUME_ARGS[@]}"
