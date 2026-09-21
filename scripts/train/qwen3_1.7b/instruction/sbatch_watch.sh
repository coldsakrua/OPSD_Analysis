#!/bin/bash
#SBATCH --job-name=watch_1p7b_instr
#SBATCH --output=log/train/qwen3_1p7b_instr/watch_slurm.%j.out
#SBATCH --partition=C64M256G,C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
set -euo pipefail

# CPU-node watcher: poll 1.7b st_tt instruction train; on ckpt-100 submit 4 think evals.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/qwen3_1p7b_instr/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}

cd "${BASE_DIR}"
mkdir -p log/train/qwen3_1p7b_instr

echo "[watch-slurm] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[watch-slurm] manifest=${MANIFEST}"
echo "[watch-slurm] interval=${INTERVAL}s post_done_wait=${POST_DONE_WAIT}s"

export BASE_DIR MANIFEST INTERVAL POST_DONE_WAIT
chmod +x "${BASE_DIR}/scripts/train/qwen3_1.7b/instruction/watch_then_eval.sh" \
  "${BASE_DIR}/scripts/train/qwen3_1.7b/instruction/submit_four_think_evals.sh"
exec bash "${BASE_DIR}/scripts/train/qwen3_1.7b/instruction/watch_then_eval.sh"
