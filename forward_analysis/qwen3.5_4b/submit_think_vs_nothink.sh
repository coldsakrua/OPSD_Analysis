#!/bin/bash
#SBATCH --job-name=fwd_tn_q35_4b
#SBATCH --output=log/forward_analysis/qwen3.5_4b/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --exclude=gpua800n13,gpua800n21
set -euo pipefail

# Qwen3.5-4B think vs nothink. Env/backend match scripts/eval/qwen3.5_4b:
#   conda qwen3_5, SGLang (triton + pytorch sampling), reasoning_parser=qwen3,
#   temp=1.0 top_p=0.95 top_k=20 presence_penalty=1.5.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen35_4b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openthoughts/train-00000-of-00002.parquet}
NUM_SAMPLES=${NUM_SAMPLES:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/forward_analysis/outputs/qwen3.5_4b/think_vs_nothink_${JOB_TAG}}

cd "${BASE_DIR}"
set +u
source activate qwen3_5
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -d /usr/local/cuda-12.8 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
elif command -v module >/dev/null 2>&1; then
  module load cuda/12.8 2>/dev/null || true
fi
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export FWD_REUSE_SAMPLES=1

mkdir -p "${OUTPUT_DIR}" "${BASE_DIR}/log/forward_analysis/qwen3.5_4b"

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --num-samples "${NUM_SAMPLES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-prompt-length 1024
  --batch-size 4
  --gen-batch-size "${GEN_BATCH_SIZE:-32}"
  --backend sglang
  --attention-backend triton
  --sampling-backend pytorch
  --mem-fraction-static 0.80
  --reasoning-parser qwen3
  --temperature 1.0
  --top-p 0.95
  --top-k 20
  --presence-penalty 1.5
  --seed 42
)

echo "[fwd] base=${BASE_DIR}"
echo "[fwd] conda=${CONDA_PREFIX}"
echo "[fwd] backend=sglang (eval-aligned)"
echo "[fwd] model=${MODEL_PATH}"
echo "[fwd] dataset=${DATASET_PATH}"
echo "[fwd] output=${OUTPUT_DIR}"
echo "[fwd] n=${NUM_SAMPLES} max_new_tokens=${MAX_NEW_TOKENS}"

echo "[fwd] ===== phase 1: generate ====="
python "${BASE_DIR}/forward_analysis/run_think_vs_nothink.py" \
  "${COMMON_ARGS[@]}" \
  --skip-activations

echo "[fwd] ===== phase 2: activations ====="
python "${BASE_DIR}/forward_analysis/run_think_vs_nothink.py" \
  "${COMMON_ARGS[@]}" \
  --skip-generate

echo "[fwd] done -> ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
