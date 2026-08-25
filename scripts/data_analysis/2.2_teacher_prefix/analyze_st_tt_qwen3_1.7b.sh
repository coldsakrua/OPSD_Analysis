#!/bin/bash
#SBATCH --job-name=da22stttqwen317b
#SBATCH --output=log/data_analysis/2.2_teacher_prefix/qwen3_1.7b/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# 2.2 teacher prefix variants (sol/answer/irr) st_tt qwen3_1.7b
# Rollout length: 1024 (override MAX_COMPLETION).
# Metrics: JSD KL, top-k KL (k=1,16), log-ratio, loss-dominant tokens.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
export TASK=teacher_prefix
export MODEL_KEY=qwen3_1.7b
export COMBO=st_tt

source "${BASE_DIR}/scripts/data_analysis/_common_launch.sh"
