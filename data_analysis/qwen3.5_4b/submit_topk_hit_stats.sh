#!/bin/bash
#SBATCH --job-name=topk_hit_q35_4b
#SBATCH --output=log/data_analysis/qwen3.5_4b/%x.%j.out
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --exclude=gpua800n13,gpua800n21
set -euo pipefail

# Top-k hit stats on existing student rollouts (sampled-token only).
# K = 4,8,16,32,64 for student & teacher under three prompt settings.
# Qwen3.5-4B: conda env qwen3_5 (NOT anchor); HF-only scoring.
# temperature=1.0 matches eval/train rollout sampling for this model.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen35_4b}
SCORE_BATCH=${SCORE_BATCH:-2}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/data_analysis/outputs/qwen3.5_4b/topk_hit_${JOB_TAG}}

if [[ -z "${ROLLOUTS_PATH:-}" ]]; then
  ROLLOUTS_PATH=$(ls -dt "${BASE_DIR}/data_analysis/outputs/qwen3.5_4b"/token_pref_*/rollouts.jsonl 2>/dev/null | head -1 || true)
fi

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
export PYTHONPATH="${BASE_DIR}/src:${BASE_DIR}/data_analysis:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}

mkdir -p "${OUTPUT_DIR}" "${BASE_DIR}/log/data_analysis/qwen3.5_4b"

if [[ -z "${ROLLOUTS_PATH:-}" || ! -f "${ROLLOUTS_PATH}" ]]; then
  echo "[error] missing rollouts under data_analysis/outputs/qwen3.5_4b/token_pref_*/" >&2
  echo "[error] run submit_token_preference_stats.sh first, or set ROLLOUTS_PATH" >&2
  exit 1
fi

echo "[topk] conda=${CONDA_PREFIX}"
echo "[topk] model=${MODEL_PATH}"
echo "[topk] rollouts=${ROLLOUTS_PATH}"
echo "[topk] output=${OUTPUT_DIR}"
echo "[topk] K=4,8,16,32,64  (sampled student token hit@k only)"

python "${BASE_DIR}/data_analysis/run_topk_hit_stats.py" \
  --model-path "${MODEL_PATH}" \
  --rollouts-path "${ROLLOUTS_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --temperature 1.0 \
  --score-batch-size "${SCORE_BATCH}" \
  --ks 4 8 16 32 64

echo "[topk] done -> ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
