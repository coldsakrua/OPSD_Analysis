#!/bin/bash
#SBATCH --job-name=sft10_opsd_1p7b
#SBATCH --output=log/train/sft10_then_opsd/1p7b_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=6:00:00
set -euo pipefail

# Pipeline: CE-SFT 10 steps → OPSD 100 steps (same 2 GPUs as baseline OPSD).
# SFT data: scripts/sft/qwen3_1.7b OT integer_answer parquet.
# OPSD hyperparams match jsd005 clip005 1e-6, but teacher has NO privilege:
#   MODE=same, TEACHER_PRIVILEGE_FIELD=none → teacher prompt == student prompt.

NUM_GPUS=2
SEED=${SEED:-42}

# --- SFT (10 steps) ---
SFT_LEARNING_RATE=${SFT_LEARNING_RATE:-1e-5}
SFT_WARMUP_RATIO=${SFT_WARMUP_RATIO:-0.03}
SFT_MAX_LENGTH=${SFT_MAX_LENGTH:-2048}
SFT_PER_DEVICE_BATCH_SIZE=${SFT_PER_DEVICE_BATCH_SIZE:-2}
SFT_GRADIENT_ACCUMULATION_STEPS=${SFT_GRADIENT_ACCUMULATION_STEPS:-16}
SFT_TARGET_GLOBAL_BATCH=${SFT_TARGET_GLOBAL_BATCH:-64}
SFT_MAX_STEPS=${SFT_MAX_STEPS:-10}
SFT_SAVE_STEPS=${SFT_SAVE_STEPS:-10}
SFT_RUN_NAME=${SFT_RUN_NAME:-sft_think_10step_ot_1p7b}

# --- OPSD (100 steps, init from SFT ckpt; no-GT same prompt) ---
MODE=${MODE:-same}
TEACHER_PRIVILEGE_FIELD=${TEACHER_PRIVILEGE_FIELD:-none}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-1024}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH:-64}
MAX_STEPS=${MAX_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-25}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.4}
OPSD_RUN_NAME=${OPSD_RUN_NAME:-st_tt_same_clip005_1e_6_ot_1p7b_sft10}

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-${BASE_MODEL_PATH}}
SFT_DATASET_PATH=${SFT_DATASET_PATH:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.opsd.solution.sthink_tthink.maxprompt1024.integer_answer.parquet}
OPSD_DATASET_PATH=${OPSD_DATASET_PATH:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.opsd.solution.sthink_tthink.maxprompt1024.parquet}
MODEL_TAG=${MODEL_TAG:-qwen3_1.7b}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/${MODEL_TAG}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-${OUTPUT_ROOT}/${SFT_RUN_NAME}/${JOB_TAG}}
OPSD_OUTPUT_DIR=${OPSD_OUTPUT_DIR:-${OUTPUT_ROOT}/${OPSD_RUN_NAME}/${JOB_TAG}}
SFT_RUN_NAME_WITH_JOB=${SFT_RUN_NAME}_${JOB_TAG}
OPSD_RUN_NAME_WITH_JOB=${OPSD_RUN_NAME}_${JOB_TAG}

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
export WANDB_DIR=${WANDB_DIR:-${BASE_DIR}/wandb}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_CONFIGURE_LOGGING=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HYDRA_FULL_ERROR=1
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "${SFT_OUTPUT_DIR}" "${OPSD_OUTPUT_DIR}" "${WANDB_DIR}" "${HF_HOME}" \
  "${BASE_DIR}/log/train/sft10_then_opsd"

for p in "${SFT_DATASET_PATH}" "${OPSD_DATASET_PATH}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[error] missing dataset: ${p}" >&2
    exit 1
  fi
done

SFT_GBS=$((SFT_PER_DEVICE_BATCH_SIZE * SFT_GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))
OPSD_GBS=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))
if [[ "${SFT_GBS}" -ne "${SFT_TARGET_GLOBAL_BATCH}" ]]; then
  echo "[error] sft global_batch=${SFT_GBS} != ${SFT_TARGET_GLOBAL_BATCH}" >&2
  exit 1
fi
if [[ "${OPSD_GBS}" -ne "${TARGET_GLOBAL_BATCH}" ]]; then
  echo "[error] opsd global_batch=${OPSD_GBS} != ${TARGET_GLOBAL_BATCH}" >&2
  exit 1
fi
if [[ "${JSD_TOKEN_CLIP}" == "none" || "${JSD_TOKEN_CLIP}" == "None" || "${JSD_TOKEN_CLIP}" == "NONE" ]]; then
  JSD_TOKEN_CLIP=0
fi

MASTER_PORT=${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}
echo "[meta] job=${JOB_TAG} model=${MODEL_TAG}"
echo "[meta] sft_out=${SFT_OUTPUT_DIR}"
echo "[meta] opsd_out=${OPSD_OUTPUT_DIR}"

# ---------- Phase 1: SFT ----------
export WANDB_PROJECT=${WANDB_PROJECT_SFT:-SFT_OpenThoughts}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP_SFT:-qwen3_1p7b_sft10_ot}
echo "[phase1] SFT steps=${SFT_MAX_STEPS} micro=${SFT_PER_DEVICE_BATCH_SIZE} gas=${SFT_GRADIENT_ACCUMULATION_STEPS} gbs=${SFT_GBS}"
accelerate launch \
  --config_file "${BASE_DIR}/configs/accelerate_zero2_no_offload.yaml" \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MASTER_PORT}" \
  "${BASE_DIR}/src/train_sft.py" \
  --model-path "${BASE_MODEL_PATH}" \
  --chat-template-path "${CHAT_TEMPLATE_PATH}" \
  --dataset-path "${SFT_DATASET_PATH}" \
  --output-dir "${SFT_OUTPUT_DIR}" \
  --run-name "${SFT_RUN_NAME_WITH_JOB}" \
  --enable-thinking \
  --max-steps "${SFT_MAX_STEPS}" \
  --save-steps "${SFT_SAVE_STEPS}" \
  --save-total-limit 2 \
  --max-length "${SFT_MAX_LENGTH}" \
  --per-device-batch-size "${SFT_PER_DEVICE_BATCH_SIZE}" \
  --gradient-accumulation-steps "${SFT_GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${SFT_LEARNING_RATE}" \
  --warmup-ratio "${SFT_WARMUP_RATIO}" \
  --seed "${SEED}" \
  --deepspeed "${BASE_DIR}/configs/deepspeed_zero2_no_offload.json"

SFT_CKPT="${SFT_OUTPUT_DIR}/checkpoint-${SFT_MAX_STEPS}"
if [[ ! -f "${SFT_CKPT}/config.json" ]]; then
  echo "[error] missing SFT checkpoint: ${SFT_CKPT}" >&2
  ls -lah "${SFT_OUTPUT_DIR}" >&2 || true
  exit 1
fi
echo "[phase1] done sft_ckpt=${SFT_CKPT}"

# ---------- Phase 2: OPSD ----------
MASTER_PORT=$((MASTER_PORT + 1))
export WANDB_PROJECT=${WANDB_PROJECT_OPSD:-OPSD}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP_OPSD:-qwen3_1p7b_sft10_then_same100}
echo "[phase2] OPSD mode=${MODE} privilege=${TEACHER_PRIVILEGE_FIELD} steps=${MAX_STEPS} init=${SFT_CKPT} lr=${LEARNING_RATE} clip=${JSD_TOKEN_CLIP} gbs=${OPSD_GBS}"
accelerate launch \
  --config_file "${BASE_DIR}/configs/accelerate_zero3.yaml" \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MASTER_PORT}" \
  "${BASE_DIR}/src/train_opsd.py" \
  --model-path "${SFT_CKPT}" \
  --dataset-path "${OPSD_DATASET_PATH}" \
  --output-dir "${OPSD_OUTPUT_DIR}" \
  --run-name "${OPSD_RUN_NAME_WITH_JOB}" \
  --privilege-mode "${MODE}" \
  --teacher-privilege-field "${TEACHER_PRIVILEGE_FIELD}" \
  --max-steps "${MAX_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --jsd-token-clip "${JSD_TOKEN_CLIP}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --deepspeed "${BASE_DIR}/configs/deepspeed_zero3.json" \
  --seed "${SEED}" \
  --student-thinking \
  --teacher-thinking

echo "[done] sft=${SFT_CKPT} opsd=${OPSD_OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
