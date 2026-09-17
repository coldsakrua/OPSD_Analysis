#!/bin/bash
#SBATCH --job-name=s4bsft15k_t4bt
#SBATCH --output=log/train/sft15000_t_think/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n13
set -euo pipefail

# Cross-init OPSD (jsd_token_clip=0.05, lr=1e-6), hyperparams match
# qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts.sh:
#   micro=4, gas=4, 2 GPU → gbs=32, max_steps=100.
# Student: Qwen3-4B-Base SFT last step (checkpoint-15000), enable_thinking=1
# Teacher: Qwen3-4B think mode (frozen), enable_thinking=1
# Teacher prompt: official OPSD template + full solution trajectory.

MODE=${MODE:-opsd}
TEACHER_PRIVILEGE_FIELD=${TEACHER_PRIVILEGE_FIELD:-solution}
STUDENT_THINKING=${STUDENT_THINKING:-1}
TEACHER_THINKING=${TEACHER_THINKING:-1}

LEARNING_RATE=${LEARNING_RATE:-1e-6}
JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}
SEED=${SEED:-42}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-1024}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH:-32}
MAX_STEPS=${MAX_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-25}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.4}

RUN_NAME=${RUN_NAME:-s4b_sft15000_t4b_think_clip005_1e_6_ot}

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
DEFAULT_SFT_CKPT=${BASE_DIR}/outputs/qwen3_4b_base/checkpoint-15000
MODEL_PATH=${MODEL_PATH:-${DEFAULT_SFT_CKPT}}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.opsd.solution.sthink_tthink.maxprompt1024.parquet}
: "${DATASET_PATH:?Set DATASET_PATH to the preprocessed OpenThoughts parquet path}"
MODEL_TAG=${MODEL_TAG:-qwen3_4b_base}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/${MODEL_TAG}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/${JOB_TAG}}
RUN_NAME_WITH_JOB=${RUN_NAME_WITH_JOB:-${RUN_NAME}_${JOB_TAG}}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}

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
export WANDB_PROJECT=${WANDB_PROJECT:-OPSD}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-sft15000_t_think_4b}
export WANDB_DIR=${WANDB_DIR:-${BASE_DIR}/wandb}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_CONFIGURE_LOGGING=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HYDRA_FULL_ERROR=1
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_HOME}" "${BASE_DIR}/log/train/sft15000_t_think"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing preprocessed dataset: ${DATASET_PATH}" >&2
  echo "[error] run: python scripts/data/preprocess_opsd_openthoughts.py --privilege-mode opsd --teacher-privilege-field solution --student-thinking --teacher-thinking --model-path ${TEACHER_MODEL_PATH} --output ${DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[error] missing student SFT checkpoint: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[error] incomplete student checkpoint (no config.json): ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${TEACHER_MODEL_PATH}" ]]; then
  echo "[error] missing teacher model: ${TEACHER_MODEL_PATH}" >&2
  exit 1
fi

THINK_ARGS=(--student-thinking --teacher-thinking)
MASTER_PORT=${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}

NUM_GPUS=2
GLOBAL_BATCH=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))
TOTAL_SAMPLES=$((GLOBAL_BATCH * MAX_STEPS))
MAX_SEQ_LEN=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))

if [[ "${GLOBAL_BATCH}" -ne "${TARGET_GLOBAL_BATCH}" ]]; then
  echo "[error] global_batch=${GLOBAL_BATCH} != TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH}" >&2
  echo "[error] keep samples/update fixed: micro * gas * gpus must equal ${TARGET_GLOBAL_BATCH}" >&2
  exit 1
fi

if [[ "${JSD_TOKEN_CLIP}" == "none" || "${JSD_TOKEN_CLIP}" == "None" || "${JSD_TOKEN_CLIP}" == "NONE" ]]; then
  JSD_TOKEN_CLIP=0
fi

echo "[launch] run=${RUN_NAME_WITH_JOB} mode=${MODE} privilege_field=${TEACHER_PRIVILEGE_FIELD}"
echo "[launch] student=${MODEL_PATH} (thinking=${STUDENT_THINKING})"
echo "[launch] teacher=${TEACHER_MODEL_PATH} (thinking=${TEACHER_THINKING})"
echo "[launch] lr=${LEARNING_RATE} jsd_token_clip=${JSD_TOKEN_CLIP} seed=${SEED}"
echo "[launch] micro=${PER_DEVICE_BATCH_SIZE} gas=${GRADIENT_ACCUMULATION_STEPS} gpus=${NUM_GPUS} → global_batch=${GLOBAL_BATCH}"
echo "[launch] max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS} → total_samples=${TOTAL_SAMPLES}"
echo "[launch] prompt=${MAX_PROMPT_LENGTH} completion=${MAX_COMPLETION_LENGTH} max_seq=${MAX_SEQ_LEN}"
echo "[launch] dataset=${DATASET_PATH} output=${OUTPUT_DIR}"
echo "[launch] master_port=${MASTER_PORT} vLLM util=${VLLM_GPU_MEMORY_UTILIZATION}"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "[launch] resume_from_checkpoint=${RESUME_FROM_CHECKPOINT} save_total_limit=${SAVE_TOTAL_LIMIT}"
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

accelerate launch \
  --config_file "${BASE_DIR}/configs/accelerate_zero3.yaml" \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MASTER_PORT}" \
  "${BASE_DIR}/src/train_opsd.py" \
  --model-path "${MODEL_PATH}" \
  --teacher-model-path "${TEACHER_MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME_WITH_JOB}" \
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
  "${RESUME_ARGS[@]}" \
  "${THINK_ARGS[@]}"
