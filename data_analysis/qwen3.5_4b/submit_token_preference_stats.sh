#!/bin/bash
#SBATCH --job-name=tokpref_q35_4b
#SBATCH --output=log/data_analysis/qwen3.5_4b/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --exclude=gpua800n13,gpua800n21
set -euo pipefail

# Token preference stats for Qwen3.5-4B.
# Env/backend match scripts/eval/qwen3.5_4b + train rollout sampling:
#   conda qwen3_5, SGLang (triton + pytorch), reasoning_parser=qwen3,
#   temp=1.0 top_p=0.95 top_k=20 presence_penalty=1.5 (nothink student).

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen35_4b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.opsd.solution.snothink_tthink.qwen35_4b.maxprompt1024.parquet}
NUM_PROMPTS=${NUM_PROMPTS:-8192}
N_ROLLOUTS=${N_ROLLOUTS:-2}
MAX_COMPLETION=${MAX_COMPLETION:-1024}
SCORE_BATCH=${SCORE_BATCH:-2}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/data_analysis/outputs/qwen3.5_4b/token_pref_${JOB_TAG}}

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
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}

mkdir -p "${OUTPUT_DIR}" "${BASE_DIR}/log/data_analysis/qwen3.5_4b"

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --num-prompts "${NUM_PROMPTS}"
  --n-rollouts "${N_ROLLOUTS}"
  --max-prompt-length 1024
  --max-completion-length "${MAX_COMPLETION}"
  --temperature 1.0
  --top-p 0.95
  --top-k 20
  --presence-penalty 1.5
  --score-batch-size "${SCORE_BATCH}"
  --gen-batch-hint "${GEN_BATCH_HINT:-64}"
  --backend sglang
  --attention-backend triton
  --sampling-backend pytorch
  --mem-fraction-static 0.80
  --reasoning-parser qwen3
  --seed 42
)

echo "[tokpref] base=${BASE_DIR}"
echo "[tokpref] conda=${CONDA_PREFIX}"
echo "[tokpref] backend=sglang (eval-aligned)"
echo "[tokpref] model=${MODEL_PATH}"
echo "[tokpref] dataset=${DATASET_PATH}"
echo "[tokpref] output=${OUTPUT_DIR}"
echo "[tokpref] prompts=${NUM_PROMPTS} × rollouts=${N_ROLLOUTS} = $((NUM_PROMPTS * N_ROLLOUTS))"
echo "[tokpref] max_completion=${MAX_COMPLETION}"

echo "[tokpref] ===== phase 1: generate ====="
python "${BASE_DIR}/data_analysis/run_token_preference_stats.py" \
  "${COMMON_ARGS[@]}" \
  --skip-score

echo "[tokpref] ===== phase 2: score ====="
python "${BASE_DIR}/data_analysis/run_token_preference_stats.py" \
  "${COMMON_ARGS[@]}" \
  --skip-generate

echo "[tokpref] done -> ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
