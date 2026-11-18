#!/bin/bash
#SBATCH --job-name=eval_first256_s42_follow
#SBATCH --output=log/eval/olmo_7b_think/array/first256_s42_follow_%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --exclude=gpua800n03,gpua800n10,gpua800n13,gpua800n21
set -euo pipefail

# Lightweight follow-up: only submits eval arrays (does not run eval itself).
# Requires TRAIN_JOBID in the environment (export via sbatch --export).
: "${TRAIN_JOBID:?Set TRAIN_JOBID}"
ROOT="${BASE_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}"
cd "${ROOT}"
bash "${ROOT}/scripts/eval/robustness/submit_olmo_first256_seed42_eval_after_train.sh"
