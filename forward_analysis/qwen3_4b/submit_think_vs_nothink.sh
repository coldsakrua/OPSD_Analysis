#!/bin/bash
#SBATCH --job-name=fwd_tn_4b
#SBATCH --output=log/forward_analysis/qwen3_4b/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openthoughts/train-00000-of-00002.parquet}
NUM_SAMPLES=${NUM_SAMPLES:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/forward_analysis/outputs/qwen3_4b/think_vs_nothink_${JOB_TAG}}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_CONFIGURE_LOGGING=0
export FWD_REUSE_SAMPLES=1

mkdir -p "${OUTPUT_DIR}" "${BASE_DIR}/log/forward_analysis/qwen3_4b"

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --num-samples "${NUM_SAMPLES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-prompt-length 1024
  --batch-size 4
  --gpu-memory-utilization 0.90
  --seed 42
)

echo "[fwd] base=${BASE_DIR}"
echo "[fwd] model=${MODEL_PATH}"
echo "[fwd] dataset=${DATASET_PATH}"
echo "[fwd] output=${OUTPUT_DIR}"
echo "[fwd] n=${NUM_SAMPLES} max_new_tokens=${MAX_NEW_TOKENS}"

# Phase 1: prepare samples + vLLM generation (own process so GPU memory is released).
echo "[fwd] ===== phase 1: generate ====="
python "${BASE_DIR}/forward_analysis/run_think_vs_nothink.py" \
  "${COMMON_ARGS[@]}" \
  --skip-activations

# Phase 2: HF activation / FFN comparison.
echo "[fwd] ===== phase 2: activations ====="
python "${BASE_DIR}/forward_analysis/run_think_vs_nothink.py" \
  "${COMMON_ARGS[@]}" \
  --skip-generate

echo "[fwd] done -> ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
