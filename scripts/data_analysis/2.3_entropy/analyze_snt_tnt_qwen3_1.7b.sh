#!/bin/bash
#SBATCH --job-name=da23_snt_tnt_1p7b
#SBATCH --output=log/data_analysis/23/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# 2.3 entropy (he20/le20/he80/le80) snt_tnt on qwen3_1.7b
# Entropy buckets: he20, le20, he80, le80 (single score pass).
# Rollout: temp=1.1 top_p=0.95 top_k=20 max_prompt=1024 max_completion=1024

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/scripts/data_analysis/outputs/entropy/qwen3_1.7b/snt_tnt_${JOB_TAG}}

TASK="entropy"
MODEL_KEY="qwen3_1.7b"
COMBO="snt_tnt"
MODEL_PATH="/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b"
CONDA_ENV="anchor"
BACKEND="vllm"
NUM_PROMPTS=${NUM_PROMPTS:-2048}
N_ROLLOUTS=${N_ROLLOUTS:-2}
MAX_PROMPT=${MAX_PROMPT:-1024}
MAX_COMPLETION=${MAX_COMPLETION:-1024}
SCORE_BATCH=${SCORE_BATCH:-8}
GEN_BATCH_HINT=${GEN_BATCH_HINT:-64}

mkdir -p "${OUTPUT_DIR}" "${BASE_DIR}/log/data_analysis/23"

cd "${BASE_DIR}"
set +u
source activate "anchor"
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export PYTHONPATH="${BASE_DIR}/src:${BASE_DIR}/scripts/data_analysis:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_CONFIGURE_LOGGING=0

EXTRA_ARGS=(
  --task "${TASK}"
  --model-key "${MODEL_KEY}"
  --combo "${COMBO}"
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --num-prompts "${NUM_PROMPTS}"
  --n-rollouts "${N_ROLLOUTS}"
  --max-prompt-length "${MAX_PROMPT}"
  --max-completion-length "${MAX_COMPLETION}"
  --temperature 1.1
  --top-p 0.95
  --top-k 20
  --score-batch-size "${SCORE_BATCH}"
  --gen-batch-hint "${GEN_BATCH_HINT}"
  --backend "${BACKEND}"
  --gpu-memory-utilization 0.90
  --seed 42
)

echo "[analysis] task=${TASK} model=${MODEL_KEY} combo=${COMBO} backend=${BACKEND}"
echo "[analysis] output=${OUTPUT_DIR}"

echo "[analysis] ===== phase 1: generate ====="
python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-score

echo "[analysis] ===== phase 2: score ====="
python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-generate

echo "[analysis] done -> ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
