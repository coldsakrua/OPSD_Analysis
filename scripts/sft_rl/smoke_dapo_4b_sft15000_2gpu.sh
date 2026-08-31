#!/bin/bash
#SBATCH --job-name=smoke_dapo_4b15k
#SBATCH --output=log/train/sft_rl/smoke_dapo_4b_sft15000_2gpu.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=12:00:00
set -euo pipefail

# 2-GPU smoke: Qwen3-4B SFT@15000 on DAPO-Math → length + math_dapo acc.
# Matches GRPO think sampling (T=1.0, top_p=0.95, top_k=20); default max=38912.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-${BASE_DIR}/outputs/qwen3_4b_base/checkpoint-15000}
CHAT_TEMPLATE_PATH=${CHAT_TEMPLATE_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/dapo/preprocessed/dapo-math-17k.qwen3.think.maxprompt1024.parquet}
NUM_SAMPLES=${NUM_SAMPLES:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-38912}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
TEMPERATURE=${TEMPERATURE:-1.0}
TP=${TP:-2}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/outputs/qwen3_4b_base/smoke_dapo_sft15000/${JOB_TAG}/summary.json}

cd "${BASE_DIR}"
mkdir -p "$(dirname "${OUTPUT_JSON}")" log/train/sft_rl

set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset PYTORCH_CUDA_ALLOC_CONF

echo "[smoke] model=${MODEL_PATH}"
echo "[smoke] dataset=${DATASET_PATH}"
echo "[smoke] n=${NUM_SAMPLES} max_new=${MAX_NEW_TOKENS} max_model_len=${MAX_MODEL_LEN} temp=${TEMPERATURE} tp=${TP}"
echo "[smoke] output=${OUTPUT_JSON}"

python -u "${BASE_DIR}/scripts/sft_rl/smoke_dapo_len_acc.py" \
  --model-path "${MODEL_PATH}" \
  --chat-template-path "${CHAT_TEMPLATE_PATH}" \
  --dataset "${DATASET_PATH}" \
  --num-samples "${NUM_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --temperature "${TEMPERATURE}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization 0.90 \
  --enable-thinking \
  --output-json "${OUTPUT_JSON}"
