#!/bin/bash
#SBATCH --exclude=gpua800n13,gpua800n21
#SBATCH --job-name=da23stttolmo37bthinkhe20
#SBATCH --output=log/data_analysis/2.3_entropy/olmo3_7b_think/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# 2.3 entropy bucket he20 st_tt olmo3_7b_think
# Rollout length: 1024 (override MAX_COMPLETION).
# Metrics: JSD KL, top-k KL (k=1,16), log-ratio, loss-dominant tokens.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
export TASK=entropy
export MODEL_KEY=olmo3_7b_think
export COMBO=st_tt
export ENTROPY_BUCKET="he20"
source "${BASE_DIR}/scripts/data_analysis/_common_launch.sh"
