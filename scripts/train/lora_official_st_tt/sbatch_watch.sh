#!/bin/bash
#SBATCH --job-name=watch_lora_st_tt
#SBATCH --output=log/train/lora_official_st_tt/watch_slurm.%j.out
#SBATCH --partition=C64M256G,C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
set -euo pipefail

# CPU-node watcher: poll official-OPSD LoRA trains; on checkpoint-100 submit 4 think evals.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/lora_official_st_tt/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_READY_WAIT=${POST_READY_WAIT:-90}

cd "${BASE_DIR}"
mkdir -p log/train/lora_official_st_tt

echo "[watch-slurm] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[watch-slurm] manifest=${MANIFEST}"
echo "[watch-slurm] interval=${INTERVAL}s post_ready_wait=${POST_READY_WAIT}s"

export BASE_DIR MANIFEST INTERVAL POST_READY_WAIT
chmod +x "${BASE_DIR}/scripts/train/lora_official_st_tt/watch_then_eval.sh" \
  "${BASE_DIR}/scripts/train/lora_official_st_tt/submit_four_evals.sh"
exec bash "${BASE_DIR}/scripts/train/lora_official_st_tt/watch_then_eval.sh"
