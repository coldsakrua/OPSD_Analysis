#!/bin/bash
#SBATCH --job-name=sft_4b_2g_smoke
#SBATCH --output=log/train/sft/4b_base/sft_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=2:00:00
set -euo pipefail

# 2-GPU smoke: micro=1, gas=8 → global_batch=16, max_steps=20, max_length=8192

NUM_GPUS=${NUM_GPUS:-2}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
MAX_LENGTH=${MAX_LENGTH:-8192}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH:-16}
MAX_STEPS=${MAX_STEPS:-20}
SAVE_STEPS=${SAVE_STEPS:-10}
RUN_NAME=${RUN_NAME:-sft_think_2gpu_smoke_8k}

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b-base}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openmathreasoning/preprocessed/omr.cot.think.le12k.all.smoke.parquet}
MODEL_TAG=${MODEL_TAG:-qwen3_4b_base}
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
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_4p4b_base_sft_smoke}
export WANDB_DIR=${WANDB_DIR:-${BASE_DIR}/wandb}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_HOME}" "${BASE_DIR}/log/train/sft/4b_base"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing dataset: ${DATASET_PATH}" >&2
  echo "[error] run: sbatch scripts/sft/qwen3_4b_base/preprocess_smoke_8k.sh" >&2
  exit 1
fi

MASTER_PORT=${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}
GLOBAL_BATCH=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))
if [[ "${GLOBAL_BATCH}" -ne "${TARGET_GLOBAL_BATCH}" ]]; then
  echo "[error] global_batch=${GLOBAL_BATCH} != TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH}" >&2
  exit 1
fi

echo "[launch] run=${RUN_NAME_WITH_JOB} gpus=${NUM_GPUS} global_batch=${GLOBAL_BATCH}"
echo "[launch] lr=${LEARNING_RATE} max_length=${MAX_LENGTH} max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS}"
echo "[launch] model=${MODEL_PATH}"
echo "[launch] dataset=${DATASET_PATH}"
echo "[launch] output=${OUTPUT_DIR} master_port=${MASTER_PORT}"

accelerate launch \
  --config_file "${BASE_DIR}/configs/accelerate_zero3.yaml" \
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
  --deepspeed "${BASE_DIR}/configs/deepspeed_zero3.json"
