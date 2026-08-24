#!/bin/bash
#SBATCH --job-name=sft_4b_8g_12k
#SBATCH --output=log/train/sft/4b_base/sft_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:8
#SBATCH --mem=800G
#SBATCH --time=360:00:00
set -euo pipefail

# 8-GPU formal (aligned with 1.7B 12k recipe total samples):
#   micro=1, gas=8 → gbs=64
#   ZeRO-2 no CPU offload (8×A800 + micro=1 @12k fits; faster ckpt than ZeRO-3)
#   max_steps=15000, save_steps=500
#   dataset: offline-tokenized le12k.all.qwen3_4b_base.tok
# Data: reuse existing omr.cot.think.le12k.all.parquet (from 1.7B pipeline); only run tokenize_le12k.sh.
# Resume (same output dir, inherit model/optim/data position):
#   RESUME_FROM_CHECKPOINT=latest OUTPUT_DIR=.../<jobid> sbatch scripts/sft/qwen3_4b_base/sft_think_8gpu_12k.sh
#   or use scripts/sft/qwen3_4b_base/sft_think_8gpu_12k_resume.sh

NUM_GPUS=${NUM_GPUS:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
MAX_LENGTH=${MAX_LENGTH:-12288}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH:-64}
MAX_STEPS=${MAX_STEPS:-15000}
SAVE_STEPS=${SAVE_STEPS:-500}
# Keep every checkpoint (do not set save_total_limit).
RUN_NAME=${RUN_NAME:-sft_think_8gpu_12k}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b-base}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openmathreasoning/preprocessed/omr.cot.think.le12k.all.qwen3_4b_base.tok}
MODEL_TAG=${MODEL_TAG:-qwen3_4b_base}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/${MODEL_TAG}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/${JOB_TAG}}
# When resuming into an existing OUTPUT_DIR, keep the original run_name (basename=old job id)
# so wandb attaches to the same curve instead of creating sft_..._<new_jobid>.
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  JOB_TAG=$(basename "${OUTPUT_DIR}")
fi
RUN_NAME_WITH_JOB=${RUN_NAME_WITH_JOB:-${RUN_NAME}_${JOB_TAG}}
WANDB_RUN_ID=${WANDB_RUN_ID:-}
WANDB_RESUME=${WANDB_RESUME:-}

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
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_4p4b_base_sft_fullparam_12k}
export WANDB_DIR=${WANDB_DIR:-${BASE_DIR}/wandb}
if [[ -n "${WANDB_RUN_ID}" ]]; then
  export WANDB_RUN_ID
fi
if [[ -n "${WANDB_RESUME}" ]]; then
  export WANDB_RESUME
elif [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  # Default: allow attaching to the previous offline/online run id.
  export WANDB_RESUME=allow
fi
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_HOME}" "${BASE_DIR}/log/train/sft/4b_base"

if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "[error] missing pretokenized dataset dir: ${DATASET_PATH}" >&2
  echo "[error] run: sbatch scripts/sft/qwen3_4b_base/tokenize_le12k.sh" >&2
  echo "[error] (reuses existing omr.cot.think.le12k.all.parquet; no raw preprocess needed)" >&2
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
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  echo "[launch] wandb_run_id=${WANDB_RUN_ID} wandb_resume=${WANDB_RESUME:-}"
elif [[ -f "${OUTPUT_DIR}/wandb_run.json" ]]; then
  echo "[launch] wandb meta=${OUTPUT_DIR}/wandb_run.json"
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  RESUME_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
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
