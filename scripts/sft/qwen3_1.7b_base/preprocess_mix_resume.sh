#!/bin/bash
#SBATCH --job-name=sft_omr_mix_cpu
#SBATCH --output=log/train/sft/1.7b_base/preprocess_%x.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=12:00:00
set -euo pipefail

# Mix/write only from already-tokenized shard_*.parquet (avoids re-tokenize after OOM).

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
RESUME_TMP_DIR=${RESUME_TMP_DIR:-${BASE_DIR}/data/openmathreasoning/preprocessed/_tmp_shards_3920179}
MID_SHORT_RATIO=${MID_SHORT_RATIO:-1:2}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""

mkdir -p "${BASE_DIR}/data/openmathreasoning/preprocessed" "${BASE_DIR}/log/train/sft/1.7b_base"

if [[ ! -d "${RESUME_TMP_DIR}" ]]; then
  echo "[preprocess] missing RESUME_TMP_DIR=${RESUME_TMP_DIR}" >&2
  exit 1
fi

echo "[preprocess] stream mix/write from ${RESUME_TMP_DIR}"
python "${BASE_DIR}/scripts/data/preprocess_sft_openmath.py" \
  --resume-tmp-dir "${RESUME_TMP_DIR}" \
  --enable-thinking \
  --max-tokens 12288 \
  --short-max-tokens 8192 \
  --mid-short-ratio "${MID_SHORT_RATIO}" \
  --keep-tmp

echo "[preprocess] done"
