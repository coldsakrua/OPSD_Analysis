#!/bin/bash
#SBATCH --job-name=tokpref_1.7b
#SBATCH --output=log/data_analysis/qwen3_1.7b/%x.%j.out
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# Token preference stats: student no-think rollouts scored by teacher-think
# under three prompt settings (opsd_sol / opsd_nogt / same).
# Default: 8192 prompts × 2 rollouts = 16384, max_completion=1024.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.opsd.solution.snothink_tthink.maxprompt1024.parquet}
NUM_PROMPTS=${NUM_PROMPTS:-8192}
N_ROLLOUTS=${N_ROLLOUTS:-2}
MAX_COMPLETION=${MAX_COMPLETION:-1024}
SCORE_BATCH=${SCORE_BATCH:-2}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/data_analysis/outputs/qwen3_1.7b/token_pref_${JOB_TAG}}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_CONFIGURE_LOGGING=0

mkdir -p "${OUTPUT_DIR}" "${BASE_DIR}/log/data_analysis/qwen3_1.7b"

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --num-prompts "${NUM_PROMPTS}"
  --n-rollouts "${N_ROLLOUTS}"
  --max-prompt-length 1024
  --max-completion-length "${MAX_COMPLETION}"
  --temperature 1.1
  --top-p 0.95
  --top-k 20
  --score-batch-size "${SCORE_BATCH}"
  --gpu-memory-utilization 0.90
  --seed 42
)

echo "[tokpref] base=${BASE_DIR}"
echo "[tokpref] model=${MODEL_PATH}"
echo "[tokpref] dataset=${DATASET_PATH}"
echo "[tokpref] output=${OUTPUT_DIR}"
echo "[tokpref] prompts=${NUM_PROMPTS} × rollouts=${N_ROLLOUTS} = $((NUM_PROMPTS * N_ROLLOUTS))"
echo "[tokpref] max_completion=${MAX_COMPLETION}"

# Phase 1: sample + vLLM generate (own process so GPU memory is released).
echo "[tokpref] ===== phase 1: generate ====="
python "${BASE_DIR}/data_analysis/run_token_preference_stats.py" \
  "${COMMON_ARGS[@]}" \
  --skip-score

# Phase 2: HF student/teacher logprobs + encourage/discourage aggregation.
echo "[tokpref] ===== phase 2: score ====="
python "${BASE_DIR}/data_analysis/run_token_preference_stats.py" \
  "${COMMON_ARGS[@]}" \
  --skip-generate

echo "[tokpref] done -> ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
