#!/bin/bash
#SBATCH --job-name=watch_c256_adv_t16
#SBATCH --output=log/train/jsd005_c256_adv_t16/watch_slurm.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
set -euo pipefail

# CPU-node watcher: wait until all 5 c256+adv_t16 trains finish, then submit
# evals while keeping >=4 GPUs free on the 16-GPU association.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/jsd005_c256_adv_t16/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-60}
GPU_QUOTA=${GPU_QUOTA:-16}
GPU_RESERVE=${GPU_RESERVE:-4}

cd "${BASE_DIR}"
mkdir -p log/train/jsd005_c256_adv_t16

echo "[watch-slurm] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[watch-slurm] manifest=${MANIFEST}"
echo "[watch-slurm] interval=${INTERVAL}s post_done_wait=${POST_DONE_WAIT}s"
echo "[watch-slurm] gpu_quota=${GPU_QUOTA} gpu_reserve=${GPU_RESERVE} cap=$((GPU_QUOTA - GPU_RESERVE))"

export BASE_DIR MANIFEST INTERVAL POST_DONE_WAIT GPU_QUOTA GPU_RESERVE
exec bash "${BASE_DIR}/scripts/train/jsd005_c256_adv_t16/watch_then_eval.sh"
