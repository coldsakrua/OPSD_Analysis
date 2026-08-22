#!/bin/bash
#SBATCH --job-name=sft_1p7b_4g_12k_r
#SBATCH --output=log/train/sft/1.7b_base/sft_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=240:00:00
set -euo pipefail

# Resume the 4-GPU 12k SFT run from the latest checkpoint in an existing output dir.
# Inherits model weights, DeepSpeed optimizer state, LR scheduler, RNG, and dataloader position.
#
# Examples:
#   sbatch scripts/sft/qwen3_1.7b_base/sft_think_4gpu_12k_resume.sh
#   OUTPUT_DIR=.../3299217 RESUME_FROM_CHECKPOINT=checkpoint-2500 sbatch ...
#   OUTPUT_DIR=.../3299217 RESUME_FROM_CHECKPOINT=latest sbatch ...

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
RUN_NAME=${RUN_NAME:-sft_think_4gpu_12k}
MODEL_TAG=${MODEL_TAG:-qwen3_1.7b_base}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/${MODEL_TAG}}

export OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/3299217}
export RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-latest}

exec bash "${BASE_DIR}/scripts/sft/qwen3_1.7b_base/sft_think_4gpu_12k.sh"
