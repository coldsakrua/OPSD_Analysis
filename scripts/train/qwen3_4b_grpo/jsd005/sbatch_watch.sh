#!/bin/bash
#SBATCH --job-name=watch_4b_grpo_opsd
#SBATCH --output=log/train/4b_grpo/watch_slurm.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
set -euo pipefail

# CPU-node watcher: poll GRPO→OPSD train; on finish submit 4 think evals
# (aime24/25/26 + hmmt25 via scripts/eval/qwen3_4b_base).

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/4b_grpo/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-60}

cd "${BASE_DIR}"
mkdir -p log/train/4b_grpo

echo "[watch-slurm] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[watch-slurm] manifest=${MANIFEST}"
echo "[watch-slurm] interval=${INTERVAL}s post_done_wait=${POST_DONE_WAIT}s"

export BASE_DIR MANIFEST INTERVAL POST_DONE_WAIT
exec bash "${BASE_DIR}/scripts/train/qwen3_4b_grpo/jsd005/watch_then_eval.sh"
