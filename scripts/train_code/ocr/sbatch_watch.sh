#!/bin/bash
#SBATCH --job-name=watch_ocr_code
#SBATCH --output=log/train_code/ocr/watch_slurm.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3906M
#SBATCH --time=7-00:00:00
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
MANIFEST=${MANIFEST:-${BASE_DIR}/log/train_code/ocr/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}
CKPT_STEP=${CKPT_STEP:-100}

cd "${BASE_DIR}"
mkdir -p log/train_code/ocr

echo "[watch-slurm] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[watch-slurm] manifest=${MANIFEST} ckpt=${CKPT_STEP}"

export BASE_DIR MANIFEST INTERVAL POST_DONE_WAIT CKPT_STEP
chmod +x "${BASE_DIR}/scripts/train_code/ocr/watch_then_eval.sh" \
  "${BASE_DIR}/scripts/train_code/ocr/submit_three_code_evals.sh"
exec bash "${BASE_DIR}/scripts/train_code/ocr/watch_then_eval.sh"
