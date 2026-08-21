#!/bin/bash
#SBATCH --job-name=sft_omr_tokenize
#SBATCH --output=log/train/sft/1.7b_base/preprocess_%x.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=122880M
#SBATCH --time=24:00:00
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
IN_PATH=${IN_PATH:-${BASE_DIR}/data/openmathreasoning/preprocessed/omr.cot.think.le12k.all.parquet}
OUT_PATH=${OUT_PATH:-${BASE_DIR}/data/openmathreasoning/preprocessed/omr.cot.think.le12k.all.tok}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b-base}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b}
NUM_PROC=${NUM_PROC:-16}
MAX_LENGTH=${MAX_LENGTH:-12288}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""

mkdir -p "${BASE_DIR}/log/train/sft/1.7b_base"

if [[ ! -f "${IN_PATH}" ]]; then
  echo "[tokenize] missing ${IN_PATH}" >&2
  exit 1
fi

echo "[tokenize] ${IN_PATH} -> ${OUT_PATH} num_proc=${NUM_PROC}"
python "${BASE_DIR}/scripts/data/tokenize_sft_le12k.py" \
  --input "${IN_PATH}" \
  --output "${OUT_PATH}" \
  --model-path "${MODEL_PATH}" \
  --chat-template-path "${CHAT_TEMPLATE_PATH}" \
  --max-length "${MAX_LENGTH}" \
  --enable-thinking \
  --num-proc "${NUM_PROC}"

echo "[tokenize] done"
