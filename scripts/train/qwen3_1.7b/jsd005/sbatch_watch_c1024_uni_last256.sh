#!/bin/bash
#SBATCH --job-name=watch_c1024_uni_last256
#SBATCH --output=log/train/watch_c1024_uni_last256.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G
#SBATCH --time=12:00:00
set -euo pipefail

# CPU-node watcher: poll uni256/last256 train jobs and submit 4 think evals each.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/c1024_uni_last256_submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-60}

cd "${BASE_DIR}"
mkdir -p log/train

echo "[watch-slurm] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[watch-slurm] manifest=${MANIFEST}"
echo "[watch-slurm] interval=${INTERVAL}s"

export MANIFEST INTERVAL POST_DONE_WAIT
exec bash "${BASE_DIR}/scripts/train/qwen3_1.7b/jsd005/watch_c1024_uni_last256_then_eval.sh"
