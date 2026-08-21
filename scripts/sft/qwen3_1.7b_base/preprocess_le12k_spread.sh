#!/bin/bash
#SBATCH --job-name=sft_omr_le12k_spread
#SBATCH --output=log/train/sft/1.7b_base/preprocess_%x.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=122880M
#SBATCH --time=12:00:00
set -euo pipefail

# Build ≤12k dataset: parallel shard convert (bounded workers) + spread_perm.npy.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
RESUME_TMP_DIR=${RESUME_TMP_DIR:-${BASE_DIR}/data/openmathreasoning/preprocessed/_tmp_shards_3920179}
OUT_PATH=${OUT_PATH:-${BASE_DIR}/data/openmathreasoning/preprocessed/omr.cot.think.le12k.all.parquet}
# 32 CPUs available, but only ~8 concurrent shard converters to avoid OOM on long CoT JSON.
WORKERS=${WORKERS:-8}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""

mkdir -p "${BASE_DIR}/data/openmathreasoning/preprocessed" "${BASE_DIR}/log/train/sft/1.7b_base"

if [[ ! -d "${RESUME_TMP_DIR}" ]]; then
  echo "[build] missing RESUME_TMP_DIR=${RESUME_TMP_DIR}" >&2
  exit 1
fi

echo "[build] smoke first (3000) workers=${WORKERS}"
python "${BASE_DIR}/scripts/data/build_sft_le12k_spread.py" \
  --resume-tmp-dir "${RESUME_TMP_DIR}" \
  --output "${OUT_PATH}" \
  --max-tokens 12288 \
  --seed 42 \
  --workers "${WORKERS}" \
  --max-samples 3000 \
  --smoke-suffix

echo "[build] full le12k.all workers=${WORKERS}"
python "${BASE_DIR}/scripts/data/build_sft_le12k_spread.py" \
  --resume-tmp-dir "${RESUME_TMP_DIR}" \
  --output "${OUT_PATH}" \
  --max-tokens 12288 \
  --seed 42 \
  --workers "${WORKERS}"

echo "[build] done"
