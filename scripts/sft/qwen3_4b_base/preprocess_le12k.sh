#!/bin/bash
#SBATCH --job-name=sft_omr_prep_cpu
#SBATCH --output=log/train/sft/4b_base/preprocess_%x.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=187488M
#SBATCH --time=24:00:00
set -euo pipefail

# CPU-only shard-wise tokenize. Peak RAM ~ workers * (shard + tokenizer), not full 3.2M table.
# C64M256G nodes have 60 CPUs; request 48 + matching DefMemPerCPU mem.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b-base}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
RAW_GLOB=${RAW_GLOB:-/gpfs/share/home/2501210611/labShare/2501210611/data/OpenMathReasoning/data/cot-*.parquet}
MID_SHORT_RATIO=${MID_SHORT_RATIO:-1:2}
SHARD_WORKERS=${SHARD_WORKERS:-40}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export CUDA_VISIBLE_DEVICES=""

mkdir -p "${BASE_DIR}/data/openmathreasoning/preprocessed" "${BASE_DIR}/log/train/sft/4b_base"

echo "[preprocess] CPU shard-wise le12k(r${MID_SHORT_RATIO}) + le8k remain"
python "${BASE_DIR}/scripts/data/preprocess_sft_openmath.py" \
  --input "${RAW_GLOB}" \
  --model-path "${MODEL_PATH}" \
  --chat-template-path "${CHAT_TEMPLATE_PATH}" \
  --enable-thinking \
  --max-tokens 12288 \
  --short-max-tokens 8192 \
  --mid-short-ratio "${MID_SHORT_RATIO}" \
  --shard-workers "${SHARD_WORKERS}"

echo "[preprocess] done"
