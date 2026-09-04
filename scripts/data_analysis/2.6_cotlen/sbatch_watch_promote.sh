#!/bin/bash
#SBATCH --job-name=da26_watch_pref
#SBATCH --output=log/data_analysis/26/%x.%j.out
#SBATCH --partition=C64M256G,C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
set -euo pipefail

# CPU-node watcher: when cotlen gen finishes, scancel leftover score phase and
# sbatch preference (submit_preference.sh). Safe to run on compute node.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
DIR="${BASE_DIR}/scripts/data_analysis/2.6_cotlen"
cd "${BASE_DIR}"
mkdir -p "${BASE_DIR}/log/data_analysis/26"

export INTERVAL="${INTERVAL:-60}"
echo "[watch-slurm] host=$(hostname) job=${SLURM_JOB_ID:-none} INTERVAL=${INTERVAL}"
exec bash "${DIR}/watch_promote_preference.sh"
