#!/bin/bash
#SBATCH --job-name=robust_watch_eval
#SBATCH --output=log/train/robustness/watch_slurm.%j.out
#SBATCH --partition=C64M256G,C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/robustness/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-60}

cd "${BASE_DIR}"
mkdir -p log/train/robustness

echo "[watch-slurm] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[watch-slurm] manifest=${MANIFEST} interval=${INTERVAL}s"

export BASE_DIR MANIFEST INTERVAL POST_DONE_WAIT
chmod +x "${BASE_DIR}/scripts/train/robustness/watch_then_eval.sh" \
  "${BASE_DIR}/scripts/train/robustness/submit_four_evals.sh"
exec bash "${BASE_DIR}/scripts/train/robustness/watch_then_eval.sh"
