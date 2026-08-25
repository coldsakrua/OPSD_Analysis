#!/bin/bash
#SBATCH --job-name=da21stttqwen34bthinking
#SBATCH --output=log/data_analysis/2.1_combinations/qwen3_4b_thinking/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# 2.1 student/teacher combo st_tt on qwen3_4b_thinking
# Rollout length: 1024 (override MAX_COMPLETION).
# Metrics: JSD KL, top-k KL (k=1,16), log-ratio, loss-dominant tokens.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
export TASK=combinations
export MODEL_KEY=qwen3_4b_thinking
export COMBO=st_tt

source "${BASE_DIR}/scripts/data_analysis/_common_launch.sh"
