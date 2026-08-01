#!/bin/bash
#SBATCH --job-name=fwd_act_q35_4b
#SBATCH --output=log/forward_analysis/qwen3.5_4b/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpua800n13,gpua800n21
set -euo pipefail

# Activations-only: paired think/nothink prompt forwards + hidden/FFN metrics.
# Faster than full generation; use to quantify representation gap first.
# Qwen3.5-4B: conda env qwen3_5 (NOT anchor); HF-only, no vLLM needed.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen35_4b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openthoughts/train-00000-of-00002.parquet}
NUM_SAMPLES=${NUM_SAMPLES:-4096}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/forward_analysis/outputs/qwen3.5_4b/activations_only_${JOB_TAG}}

cd "${BASE_DIR}"
set +u
source activate qwen3_5
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Prefer CUDA 12.8 (matches qwen3_5 / torch cu128).
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

mkdir -p "${OUTPUT_DIR}" "${BASE_DIR}/log/forward_analysis/qwen3.5_4b"

echo "[fwd-act] conda=${CONDA_PREFIX}"
echo "[fwd-act] output=${OUTPUT_DIR} n=${NUM_SAMPLES}"
python "${BASE_DIR}/forward_analysis/run_think_vs_nothink.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-samples "${NUM_SAMPLES}" \
  --max-new-tokens 2048 \
  --max-prompt-length 1024 \
  --batch-size 4 \
  --seed 42 \
  --skip-generate

echo "[fwd-act] done"
ls -lah "${OUTPUT_DIR}"
python - <<PY
import json
from pathlib import Path
p = Path("${OUTPUT_DIR}") / "summary.json"
print(p.read_text())
PY
