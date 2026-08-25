#!/bin/bash
#SBATCH --exclude=gpua800n13,gpua800n21
#SBATCH --job-name=da22snttntolmo37binstruct
#SBATCH --output=log/data_analysis/2.2_teacher_prefix/olmo3_7b_instruct/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# 2.2 teacher prefix variants (sol/answer/irr) snt_tnt olmo3_7b_instruct
# Rollout length: 1024 (override MAX_COMPLETION).
# Metrics: JSD KL, top-k KL (k=1,16), log-ratio, loss-dominant tokens.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
export TASK=teacher_prefix
export MODEL_KEY=olmo3_7b_instruct
export COMBO=snt_tnt

source "${BASE_DIR}/scripts/data_analysis/_common_launch.sh"
