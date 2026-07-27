#!/bin/bash
#SBATCH --job-name=snt_tnt_1e6_ot
#SBATCH --output=log/train/4b/opsd_%x.%j.out
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n03,gpua800n09,gpua800n13
set -euo pipefail

# Same as opsd_student_nothink_teacher_think_4b_1e_6_openthoughts.sh, but:
# - student AND teacher are both nothink
# - teacher privilege = full solution trajectory
# - lr=1e-6
MODE=opsd
TEACHER_PRIVILEGE_FIELD=solution
STUDENT_THINKING=0
TEACHER_THINKING=0
RUN_NAME=snt_tnt_1e_6_openthoughts
LEARNING_RATE=1e-6

# Slurm copies the batch script to /var/spool; prefer submit dir over BASH_SOURCE.
BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
# Offline length-filtered for student TM-off + teacher TM-off + full-solution privilege.
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.opsd.solution.nothink.maxprompt1024.parquet}
: "${DATASET_PATH:?Set DATASET_PATH to the preprocessed OpenThoughts parquet path}"
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs}
# One folder per run (Slurm job id); avoid overwriting previous checkpoints.
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/${JOB_TAG}}
RUN_NAME_WITH_JOB=${RUN_NAME}_${JOB_TAG}

cd "${BASE_DIR}"
# conda activate scripts reference unset vars; keep nounset elsewhere
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
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_4b_fullparam_100step_openthoughts}
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
  echo "[error] missing preprocessed dataset: ${DATASET_PATH}" >&2
  echo "[error] run: python scripts/data/preprocess_opsd_openthoughts.py --privilege-mode opsd --teacher-privilege-field solution --no-student-thinking --no-teacher-thinking" >&2
  exit 1
fi

THINK_ARGS=(--no-student-thinking --no-teacher-thinking)

# Avoid fixed 29500 collisions when multiple train jobs share a node.
MASTER_PORT=${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}

echo "[launch] run=${RUN_NAME_WITH_JOB} mode=${MODE} privilege_field=${TEACHER_PRIVILEGE_FIELD} student_thinking=${STUDENT_THINKING} teacher_thinking=${TEACHER_THINKING} lr=${LEARNING_RATE}"
echo "[launch] model=${MODEL_PATH} dataset=${DATASET_PATH} output=${OUTPUT_DIR}"
echo "[launch] master_port=${MASTER_PORT} 2 GPUs, microbatch=4, gas=4, global batch=32, vLLM util=0.4, gen-once-per-step"

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
