#!/bin/bash
#SBATCH --job-name=smoke_apps_ext
#SBATCH --output=log/train_code/sft_rl/smoke_apps_extract.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
set -euo pipefail

# 1-GPU smoke: OCR SFT generates on APPS → extract + reward.
# Sampling matches GRPO (scripts/sft_rl/common.sh): T=0.6, top_p=0.95, top_k=20.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-${BASE_DIR}/outputs/qwen3_4b_base/sft_think_4gpu_12k_ocr/3925962/checkpoint-15000}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/apps/preprocessed/apps.traintest.qwen3.think.maxprompt2048.maxtests10.parquet}
NUM_SAMPLES=${NUM_SAMPLES:-64}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-24576}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-26624}
# Align with GRPO common.sh
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
TP=${TP:-1}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/outputs/qwen3_4b_base/smoke_apps_extract_ocr/${JOB_TAG}/summary.json}

cd "${BASE_DIR}"
mkdir -p "$(dirname "${OUTPUT_JSON}")" log/train_code/sft_rl

set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/src:${BASE_DIR}/eval_code:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_LOGGING_LEVEL=ERROR
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset PYTORCH_CUDA_ALLOC_CONF

echo "[smoke] model=${MODEL_PATH}"
echo "[smoke] dataset=${DATASET_PATH}"
echo "[smoke] problems=${NUM_SAMPLES} n_gen=${NUM_GENERATIONS} total=$((NUM_SAMPLES * NUM_GENERATIONS))"
echo "[smoke] max_new=${MAX_NEW_TOKENS} max_model_len=${MAX_MODEL_LEN} temp=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K} tp=${TP}"
echo "[smoke] output=${OUTPUT_JSON}"

python -u "${BASE_DIR}/scripts/train_code/sft_rl/smoke_apps_extract.py" \
  --model-path "${MODEL_PATH}" \
  --chat-template-path "${CHAT_TEMPLATE_PATH}" \
  --dataset "${DATASET_PATH}" \
  --num-samples "${NUM_SAMPLES}" \
  --num-generations "${NUM_GENERATIONS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization 0.90 \
  --enable-thinking \
  --quiet-rows \
  --output-json "${OUTPUT_JSON}"
