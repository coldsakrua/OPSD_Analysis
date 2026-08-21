#!/bin/bash
#SBATCH --job-name=sft_omr_smoke_cpu
#SBATCH --output=log/train/sft/1.7b_base/preprocess_%x.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=02:00:00
set -euo pipefail

# CPU-only smoke preprocess (one CoT shard).

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b-base}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b}
RAW_GLOB=${RAW_GLOB:-/gpfs/share/home/2501210611/labShare/2501210611/data/OpenMathReasoning/data/cot-00000-of-00144.parquet}
MID_SHORT_RATIO=${MID_SHORT_RATIO:-1:2}
MAX_SAMPLES=${MAX_SAMPLES:-3000}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export CUDA_VISIBLE_DEVICES=""

mkdir -p "${BASE_DIR}/data/openmathreasoning/preprocessed" "${BASE_DIR}/log/train/sft/1.7b_base"

echo "[preprocess] CPU smoke from ${RAW_GLOB}"
python "${BASE_DIR}/scripts/data/preprocess_sft_openmath.py" \
  --input "${RAW_GLOB}" \
  --model-path "${MODEL_PATH}" \
  --chat-template-path "${CHAT_TEMPLATE_PATH}" \
  --enable-thinking \
  --max-tokens 12288 \
  --short-max-tokens 8192 \
  --mid-short-ratio "${MID_SHORT_RATIO}" \
  --smoke \
  --max-samples "${MAX_SAMPLES}" \
  --shard-workers 2

echo "[preprocess] done"
