#!/bin/bash
#SBATCH --job-name=sft_1p7b_ocr_12k_r
#SBATCH --output=log/train_code/sft/1.7b_base/sft_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=240:00:00
set -euo pipefail

#   OUTPUT_DIR=.../<jobid> sbatch scripts/sft_code/qwen3_1.7b_base/sft_think_4gpu_12k_ocr_resume.sh

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
RUN_NAME=${RUN_NAME:-sft_think_4gpu_12k_ocr}
MODEL_TAG=${MODEL_TAG:-qwen3_1.7b_base}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/${MODEL_TAG}}

export OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR to the existing run directory}
export RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-latest}
export WANDB_RESUME=${WANDB_RESUME:-allow}
export RUN_NAME_WITH_JOB=${RUN_NAME_WITH_JOB:-${RUN_NAME}_$(basename "${OUTPUT_DIR}")}

if [[ -z "${WANDB_RUN_ID:-}" && -f "${OUTPUT_DIR}/wandb_run.json" ]]; then
  WANDB_RUN_ID=$(python3 - <<PY
import json
print(json.load(open("${OUTPUT_DIR}/wandb_run.json"))["id"])
PY
)
  export WANDB_RUN_ID
fi

exec bash "${BASE_DIR}/scripts/sft_code/qwen3_1.7b_base/sft_think_4gpu_12k_ocr.sh"
