#!/bin/bash
#SBATCH --job-name=watch_olmo_instrrev
#SBATCH --output=log/train/olmo3_7b_think_instrrev/watch_slurm.%j.out
#SBATCH --partition=C64M256G,C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
set -euo pipefail

# CPU-node watcher: poll Olmo st_tt instruction_rev train; on ckpt-100 submit 4 think evals.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/olmo3_7b_think_instrrev/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}

cd "${BASE_DIR}"
mkdir -p log/train/olmo3_7b_think_instrrev

echo "[watch-slurm] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[watch-slurm] manifest=${MANIFEST}"
echo "[watch-slurm] interval=${INTERVAL}s post_done_wait=${POST_DONE_WAIT}s"

export BASE_DIR MANIFEST INTERVAL POST_DONE_WAIT
chmod +x "${BASE_DIR}/scripts/train/olmo3_7b_think/instruction_rev/watch_then_eval.sh" \
  "${BASE_DIR}/scripts/train/olmo3_7b_think/instruction_rev/submit_four_think_evals.sh"
exec bash "${BASE_DIR}/scripts/train/olmo3_7b_think/instruction_rev/watch_then_eval.sh"
