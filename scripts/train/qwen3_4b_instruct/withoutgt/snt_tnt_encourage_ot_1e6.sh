#!/bin/bash
#SBATCH --job-name=snt_tnt_enc_oti
#SBATCH --output=log/train/4b-instruct/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n02,gpua800n03,gpua800n04,gpua800n05,gpua800n06,gpua800n09,gpua800n10,gpua800n13,gpua800n14
set -euo pipefail

# Instruct counterpart of qwen3_4b/withoutgt/snt_tt_encourage_ot_1e6.sh:
# - Instruct has no native think mode; thinking flags always off
# - Teacher: English encouragement (no solution/answer) + same student user content
MODE=encourage
TEACHER_PRIVILEGE_FIELD=none
STUDENT_THINKING=0
TEACHER_THINKING=0
RUN_NAME=snt_tnt_encourage_ot_1e_6_instruct
LEARNING_RATE=1e-6

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b-instruct}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.correct.answer.nothink.instruct.maxprompt1024.parquet}
: "${DATASET_PATH:?Set DATASET_PATH to an OpenThoughts parquet with a problem column}"
MODEL_TAG=${MODEL_TAG:-qwen3_4b_instruct}
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
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_4b_instruct_withoutgt_encourage}
export WANDB_DIR=${WANDB_DIR:-${BASE_DIR}/wandb}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_CONFIGURE_LOGGING=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HYDRA_FULL_ERROR=1
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_HOME}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing dataset: ${DATASET_PATH}" >&2
  exit 1
fi

THINK_ARGS=(--no-student-thinking --no-teacher-thinking)
MASTER_PORT=${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}

echo "[launch] run=${RUN_NAME_WITH_JOB} mode=${MODE} privilege_field=${TEACHER_PRIVILEGE_FIELD} thinking=off lr=${LEARNING_RATE}"
echo "[launch] model=${MODEL_PATH} dataset=${DATASET_PATH} output=${OUTPUT_DIR}"
echo "[launch] master_port=${MASTER_PORT} 2 GPUs, microbatch=4, gas=4, global batch=32, vLLM util=0.4"

accelerate launch \
  --config_file "${BASE_DIR}/configs/accelerate_zero3.yaml" \
  --num_processes 2 \
  --main_process_port "${MASTER_PORT}" \
  "${BASE_DIR}/src/train_opsd.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME_WITH_JOB}" \
  --privilege-mode "${MODE}" \
  --teacher-privilege-field "${TEACHER_PRIVILEGE_FIELD}" \
  --max-steps 100 \
  --save-steps 25 \
  --max-prompt-length 1024 \
  --max-completion-length 1024 \
  --per-device-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate "${LEARNING_RATE}" \
  --vllm-gpu-memory-utilization 0.4 \
  --deepspeed "${BASE_DIR}/configs/deepspeed_zero3.json" \
  "${THINK_ARGS[@]}"
