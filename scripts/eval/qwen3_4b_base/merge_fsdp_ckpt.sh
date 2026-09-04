#!/bin/bash
#SBATCH --job-name=merge_grpo4b_s100
#SBATCH --output=log/eval/qwen3_4b_base/merge/%x.%j.out
#SBATCH --partition=C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

LOCAL_DIR=${LOCAL_DIR:?}
TARGET_DIR=${TARGET_DIR:?}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "$(dirname "${TARGET_DIR}")" "log/eval/qwen3_4b_base/merge"

if [[ -f "${TARGET_DIR}/config.json" ]] && compgen -G "${TARGET_DIR}/*.safetensors" >/dev/null; then
  echo "[merge] already exists: ${TARGET_DIR}"
  ls -lah "${TARGET_DIR}" | head -20
  exit 0
fi

echo "[merge] local_dir=${LOCAL_DIR}"
echo "[merge] target_dir=${TARGET_DIR}"

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${LOCAL_DIR}" \
  --target_dir "${TARGET_DIR}" \
  --use_cpu_initialization

echo "[merge] done"
ls -lah "${TARGET_DIR}" | head -30
