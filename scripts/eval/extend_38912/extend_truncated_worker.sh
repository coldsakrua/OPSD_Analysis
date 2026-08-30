#!/bin/bash
#SBATCH --job-name=ext38912
#SBATCH --output=log/eval/extend_38912/worker/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# One SLURM job processes a shard jsonl that may contain multiple models/datasets.
# Usage (from BASE_DIR):
#   SHARD_FILE=scripts/eval/extend_38912/manifest/shards/shard_00.jsonl \
#     sbatch scripts/eval/extend_38912/extend_truncated_worker.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

SHARD_FILE=${SHARD_FILE:?Set SHARD_FILE to a shard_XX.jsonl}
# Allow relative shard paths from BASE_DIR.
if [[ "${SHARD_FILE}" != /* ]]; then
  SHARD_FILE="${BASE_DIR}/${SHARD_FILE}"
fi
GENERATE_BATCH_SIZE=${GENERATE_BATCH_SIZE:-0}  # 0 = auto by model size
GPU_MEM=${GPU_MEM:-0.9}
# Set ALLOW_LONG_MAX_MODEL_LEN=1 for qwen3-*-base (max_pos=32768) extending to 40k.
ALLOW_LONG_MAX_MODEL_LEN=${ALLOW_LONG_MAX_MODEL_LEN:-0}

cd "${BASE_DIR}"
mkdir -p log/eval/extend_38912/worker

set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/vendor/verl:${BASE_DIR}/eval:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
if [[ "${ALLOW_LONG_MAX_MODEL_LEN}" == "1" ]]; then
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi

echo "[extend] base_dir=${BASE_DIR}"
echo "[extend] shard=${SHARD_FILE}"
echo "[extend] generate_batch_size=${GENERATE_BATCH_SIZE} (0=auto: 0.6b/1.7b=12, 4b=8, 8b=4)"
echo "[extend] allow_long_max_model_len=${ALLOW_LONG_MAX_MODEL_LEN}"
echo "[extend] conda_prefix=${CONDA_PREFIX}"

python "${BASE_DIR}/eval/extend_truncated_vllm.py" \
  --shard-file "${SHARD_FILE}" \
  --generate-batch-size "${GENERATE_BATCH_SIZE}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEM}"
