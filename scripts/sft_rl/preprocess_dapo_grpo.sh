#!/bin/bash
#SBATCH --job-name=prep_dapo_grpo
#SBATCH --output=log/train/sft_rl/preprocess_dapo_grpo.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-${MODEL_PATH}}
INPUT=${INPUT:-${BASE_DIR}/data/dapo/dapo-math-17k.parquet}
OUTPUT=${OUTPUT:-${BASE_DIR}/data/dapo/preprocessed/dapo-math-17k.qwen3.think.maxprompt1024.parquet}

mkdir -p "${BASE_DIR}/log/train/sft_rl" "$(dirname "${OUTPUT}")"

python "${BASE_DIR}/scripts/data/preprocess_dapo_grpo.py" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --model-path "${MODEL_PATH}" \
  --chat-template-path "${CHAT_TEMPLATE_PATH}" \
  --max-prompt-length 1024 \
  --enable-thinking
