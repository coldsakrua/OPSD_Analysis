#!/bin/bash
#SBATCH --job-name=pmi_watch_eval
#SBATCH --output=log/train/purified_pmi/%x.%j.out
#SBATCH --partition=C64M256G,C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
set -euo pipefail

# Compute-node (CPU) watcher: when each Purified-OPSD LoRA train finishes,
# submit 4 evals (aime24/25/26 + hmmt25).

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
DIR="${BASE_DIR}/scripts/train/purified_pmi"
cd "${BASE_DIR}"
mkdir -p "${BASE_DIR}/log/train/purified_pmi"

export INTERVAL="${INTERVAL:-120}"
echo "[watch-slurm] host=$(hostname) job=${SLURM_JOB_ID:-none} INTERVAL=${INTERVAL}"
chmod +x "${DIR}/watch_and_eval.sh" "${DIR}/submit_four_evals.sh" "${DIR}/merge_olmo_lora.sh"
exec bash "${DIR}/watch_and_eval.sh"
