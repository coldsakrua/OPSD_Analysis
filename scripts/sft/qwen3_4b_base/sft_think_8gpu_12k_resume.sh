#!/bin/bash
#SBATCH --job-name=sft_4b_8g_12k_r
#SBATCH --output=log/train/sft/4b_base/sft_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:8
#SBATCH --mem=800G
#SBATCH --time=360:00:00
set -euo pipefail

# Resume the 8-GPU 12k SFT run from the latest checkpoint in an existing output dir.
# Inherits model/optim/scheduler/RNG/dataloader position, and continues the same wandb curve.
#
# Examples:
#   OUTPUT_DIR=.../<jobid> sbatch scripts/sft/qwen3_4b_base/sft_think_8gpu_12k_resume.sh
#   OUTPUT_DIR=.../<jobid> RESUME_FROM_CHECKPOINT=checkpoint-3500 sbatch ...
#   WANDB_RUN_ID=<id> OUTPUT_DIR=.../<jobid> sbatch ...

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
RUN_NAME=${RUN_NAME:-sft_think_8gpu_12k}
MODEL_TAG=${MODEL_TAG:-qwen3_4b_base}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/${MODEL_TAG}}

: "${OUTPUT_DIR:?Set OUTPUT_DIR to the existing run directory (e.g. outputs/qwen3_4b_base/sft_think_8gpu_12k/<jobid>)}"
export RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-latest}
export WANDB_RESUME=${WANDB_RESUME:-allow}

# Keep original wandb display name (not the new SLURM job id).
export RUN_NAME_WITH_JOB=${RUN_NAME_WITH_JOB:-${RUN_NAME}_$(basename "${OUTPUT_DIR}")}

# Prefer explicit env, else output_dir/wandb_run.json (written by train_sft / seeded below).
if [[ -z "${WANDB_RUN_ID:-}" && -f "${OUTPUT_DIR}/wandb_run.json" ]]; then
  WANDB_RUN_ID=$(python3 - <<PY
import json
print(json.load(open("${OUTPUT_DIR}/wandb_run.json"))["id"])
PY
)
  export WANDB_RUN_ID
fi

exec bash "${BASE_DIR}/scripts/sft/qwen3_4b_base/sft_think_8gpu_12k.sh"
