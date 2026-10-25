#!/bin/bash
#SBATCH --job-name=watch_ocr_r300
#SBATCH --output=log/train_code/ocr_extend300/watch_slurm.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3906M
#SBATCH --time=7-00:00:00
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
MANIFEST=${MANIFEST:-${BASE_DIR}/log/train_code/ocr_extend300/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}

cd "${BASE_DIR}"
mkdir -p log/train_code/ocr_extend300

echo "[watch-slurm] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[watch-slurm] manifest=${MANIFEST}"

export BASE_DIR MANIFEST INTERVAL POST_DONE_WAIT
chmod +x "${BASE_DIR}/scripts/train_code/ocr_extend300/watch_then_eval_multi.sh" \
  "${BASE_DIR}/scripts/train_code/ocr/submit_three_code_evals.sh"
exec bash "${BASE_DIR}/scripts/train_code/ocr_extend300/watch_then_eval_multi.sh"
