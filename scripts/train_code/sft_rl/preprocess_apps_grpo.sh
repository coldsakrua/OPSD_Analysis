#!/bin/bash
#SBATCH --job-name=prep_apps_grpo
#SBATCH --output=log/train_code/sft_rl/preprocess_apps_grpo.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
set -euo pipefail

# Build APPS GRPO parquet: train+test, all difficulties.
# Call-based (introductory) outputs are unwrapped for LCB judge compatibility.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-${MODEL_PATH}}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_TESTS=${MAX_TESTS:-10}
SPLIT=${SPLIT:-train+test}

mkdir -p "${BASE_DIR}/log/train_code/sft_rl" "${BASE_DIR}/data/apps/preprocessed"

echo "[prep] split=${SPLIT} max_prompt=${MAX_PROMPT_LENGTH} max_tests=${MAX_TESTS}"

python -u "${BASE_DIR}/scripts/data/preprocess_apps_grpo.py" \
  --split "${SPLIT}" \
  --model-path "${MODEL_PATH}" \
  --chat-template-path "${CHAT_TEMPLATE_PATH}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH}" \
  --max-tests "${MAX_TESTS}" \
  --enable-thinking

# Small held-out slice from test only (optional smoke/val); skip if SPLIT is already train+test-only need.
# Keep a separate short test parquet for optional val references.
python -u "${BASE_DIR}/scripts/data/preprocess_apps_grpo.py" \
  --split test \
  --model-path "${MODEL_PATH}" \
  --chat-template-path "${CHAT_TEMPLATE_PATH}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH}" \
  --max-tests "${MAX_TESTS}" \
  --enable-thinking \
  --limit "${TEST_LIMIT:-200}"

echo "[prep] done"
