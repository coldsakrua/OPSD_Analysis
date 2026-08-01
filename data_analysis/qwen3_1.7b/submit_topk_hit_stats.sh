#!/bin/bash
#SBATCH --job-name=topk_hit_1.7b
#SBATCH --output=log/data_analysis/qwen3_1.7b/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# Top-k hit stats on existing student rollouts (sampled-token only).
# K = 4,8,16,32,64 for student & teacher under three prompt settings.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b}
SCORE_BATCH=${SCORE_BATCH:-2}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/data_analysis/outputs/qwen3_1.7b/topk_hit_${JOB_TAG}}

if [[ -z "${ROLLOUTS_PATH:-}" ]]; then
  ROLLOUTS_PATH=$(ls -dt "${BASE_DIR}/data_analysis/outputs/qwen3_1.7b"/token_pref_*/rollouts.jsonl 2>/dev/null | head -1 || true)
fi

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/src:${BASE_DIR}/data_analysis:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}

mkdir -p "${OUTPUT_DIR}" "${BASE_DIR}/log/data_analysis/qwen3_1.7b"

if [[ -z "${ROLLOUTS_PATH:-}" || ! -f "${ROLLOUTS_PATH}" ]]; then
  echo "[error] missing rollouts under data_analysis/outputs/qwen3_1.7b/token_pref_*/" >&2
  echo "[error] run submit_token_preference_stats.sh first, or set ROLLOUTS_PATH" >&2
  exit 1
fi

echo "[topk] model=${MODEL_PATH}"
echo "[topk] rollouts=${ROLLOUTS_PATH}"
echo "[topk] output=${OUTPUT_DIR}"
echo "[topk] K=4,8,16,32,64  (sampled student token hit@k only)"

python "${BASE_DIR}/data_analysis/run_topk_hit_stats.py" \
  --model-path "${MODEL_PATH}" \
  --rollouts-path "${ROLLOUTS_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --temperature 1.1 \
  --score-batch-size "${SCORE_BATCH}" \
  --ks 4 8 16 32 64

echo "[topk] done -> ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
