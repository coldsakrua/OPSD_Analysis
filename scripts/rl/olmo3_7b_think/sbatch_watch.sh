#!/bin/bash
#SBATCH --job-name=watch_grpo_olmo
#SBATCH --output=log/train/rl/olmo3_7b_think/watch_slurm.%j.out
#SBATCH --partition=C64M256G,C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
set -euo pipefail

# CPU watcher for Olmo-3-7B-Think GRPO: poll train → merge FSDP → 4 think evals.
# Train walltime is 120h; keep watch longer so gs100 is not missed.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/rl/olmo3_7b_think/watch_grpo_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-120}

cd "${BASE_DIR}"
mkdir -p log/train/rl/olmo3_7b_think

echo "[watch-slurm] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[watch-slurm] manifest=${MANIFEST}"
echo "[watch-slurm] interval=${INTERVAL}s post_done_wait=${POST_DONE_WAIT}s"

export BASE_DIR MANIFEST INTERVAL POST_DONE_WAIT
chmod +x "${BASE_DIR}/scripts/rl/olmo3_7b_think/watch_then_eval.sh"
exec bash "${BASE_DIR}/scripts/rl/olmo3_7b_think/watch_then_eval.sh"
