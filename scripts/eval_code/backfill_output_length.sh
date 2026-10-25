#!/bin/bash
#SBATCH --job-name=backfill_olen
#SBATCH --output=log/eval_code/_backfill/%x.%j.out
#SBATCH --partition=C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#
# CPU backfill: avg_output_tokens for finished code-eval generations → metrics + logs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${BASE_DIR}"
mkdir -p log/eval_code/_backfill

CONDA_ROOT="${CONDA_ROOT:-/gpfs/share/home/2501210611/miniconda3}"
PY="${CONDA_ROOT}/envs/${CONDA_ENV:-anchor}/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "[backfill] missing python: ${PY}" >&2
  exit 1
fi

echo "[backfill] host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} base=${BASE_DIR}"
echo "[backfill] python=${PY}"
echo "[backfill] start=$(date -Is)"
"${PY}" -u scripts/eval_code/backfill_output_length.py "$@"
echo "[backfill] end=$(date -Is)"
