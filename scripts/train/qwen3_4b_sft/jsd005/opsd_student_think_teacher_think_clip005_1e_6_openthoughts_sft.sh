#!/bin/bash
#SBATCH --job-name=st_tt_clip005_1e6_4bs
#SBATCH --output=log/train/4b_sft/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=72:00:00
set -euo pipefail

# Qwen3-4B-Base SFT checkpoint OPSD (st_tt): student think + teacher think.
# Hyperparams match qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts.sh:
#   jsd_token_clip=0.05, lr=1e-6, micro=4, gas=4, 2 GPU → gbs=32, max_steps=100.
# Default MODEL_PATH: SFT checkpoint (override with env MODEL_PATH=.../checkpoint-N).

MODE=${MODE:-opsd}
TEACHER_PRIVILEGE_FIELD=${TEACHER_PRIVILEGE_FIELD:-solution}
STUDENT_THINKING=${STUDENT_THINKING:-1}
TEACHER_THINKING=${TEACHER_THINKING:-1}

LEARNING_RATE=${LEARNING_RATE:-1e-6}
JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
MAX_STEPS=${MAX_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-25}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.4}

RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_openthoughts_4b_sft}

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}}
# Override MODEL_PATH after SFT run completes, e.g.:
#   MODEL_PATH=outputs/qwen3_4b_base/sft_think_8gpu_12k/<jobid>/checkpoint-15000 sbatch ...
MODEL_PATH=${MODEL_PATH:-}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.opsd.solution.sthink_tthink.maxprompt1024.parquet}
: "${DATASET_PATH:?Set DATASET_PATH to the preprocessed OpenThoughts parquet path}"
: "${MODEL_PATH:?Set MODEL_PATH to an SFT checkpoint directory}"
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
export WANDB_PROJECT=${WANDB_PROJECT:-OPSD}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_4b_sft_fullparam_100step_openthoughts}
export WANDB_DIR=${WANDB_DIR:-${BASE_DIR}/wandb}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_CONFIGURE_LOGGING=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HYDRA_FULL_ERROR=1
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_HOME}" "${BASE_DIR}/log/train/4b_sft"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing preprocessed dataset: ${DATASET_PATH}" >&2
  echo "[error] run: python scripts/data/preprocess_opsd_openthoughts.py --privilege-mode opsd --teacher-privilege-field solution --student-thinking --teacher-thinking --model-path /path/to/qwen3-4b --output ${DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[error] missing SFT checkpoint: ${MODEL_PATH}" >&2
  echo "[error] set MODEL_PATH=.../checkpoint-N to your 4B SFT run output" >&2
  exit 1
fi

THINK_ARGS=(--student-thinking --teacher-thinking)
MASTER_PORT=${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}

NUM_GPUS=2
GLOBAL_BATCH=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))

if [[ "${JSD_TOKEN_CLIP}" == "none" || "${JSD_TOKEN_CLIP}" == "None" || "${JSD_TOKEN_CLIP}" == "NONE" ]]; then
  JSD_TOKEN_CLIP=0
fi

echo "[launch] run=${RUN_NAME_WITH_JOB} mode=${MODE} privilege_field=${TEACHER_PRIVILEGE_FIELD}"
echo "[launch] student_thinking=${STUDENT_THINKING} teacher_thinking=${TEACHER_THINKING}"
echo "[launch] lr=${LEARNING_RATE} jsd_token_clip=${JSD_TOKEN_CLIP}"
echo "[launch] micro=${PER_DEVICE_BATCH_SIZE} gas=${GRADIENT_ACCUMULATION_STEPS} gpus=${NUM_GPUS} → global_batch=${GLOBAL_BATCH}"
echo "[launch] max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS}"
echo "[launch] model=${MODEL_PATH} dataset=${DATASET_PATH} output=${OUTPUT_DIR}"
echo "[launch] master_port=${MASTER_PORT} vLLM util=${VLLM_GPU_MEMORY_UTILIZATION}"

accelerate launch \
  --config_file "${BASE_DIR}/configs/accelerate_zero3.yaml" \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MASTER_PORT}" \
  "${BASE_DIR}/src/train_opsd.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME_WITH_JOB}" \
  --privilege-mode "${MODE}" \
  --teacher-privilege-field "${TEACHER_PRIVILEGE_FIELD}" \
  --max-steps "${MAX_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --max-prompt-length 1024 \
  --max-completion-length 1024 \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --jsd-token-clip "${JSD_TOKEN_CLIP}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --deepspeed "${BASE_DIR}/configs/deepspeed_zero3.json" \
  "${THINK_ARGS[@]}"
