#!/bin/bash
#SBATCH --job-name=da22omr_st_tt_0p6b
#SBATCH --output=log/data_analysis/22_omr/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=3:00:00
set -euo pipefail

# OMR full-solution teacher_prefix (st_tt / qwen3_06b)
# Shared 2048 problems × 2 rollouts; privilege=solution (opsd).
# Teacher prompt ≤2048, completion ≤1024 → seq ≤3072.
# SCORE_BATCH/GEN_BATCH calibrated for 2k teacher prompts on A800 80GB.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
RUN_SUFFIX="omr_solution"
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/scripts/data_analysis/outputs/teacher_prefix_omr/qwen3_06b/${RUN_SUFFIX}_${JOB_TAG}}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openmathreasoning/preprocessed/omr.opsd.solution.postthink.shared2048.seed42.maxprompt2048.parquet}

TASK="teacher_prefix"
MODEL_KEY="qwen3_06b"
COMBO="st_tt"
MODEL_PATH="/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-0.6b"
CONDA_ENV="anchor"
BACKEND="vllm"
NUM_PROMPTS=${NUM_PROMPTS:-2048}
N_ROLLOUTS=${N_ROLLOUTS:-2}
MAX_PROMPT=${MAX_PROMPT:-2048}
MAX_COMPLETION=${MAX_COMPLETION:-1024}
SCORE_BATCH=${SCORE_BATCH:-8}
GEN_BATCH_HINT=${GEN_BATCH_HINT:-256}

mkdir -p "${OUTPUT_DIR}" "${BASE_DIR}/log/data_analysis/22_omr"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing shared OMR parquet: ${DATASET_PATH}" >&2
  echo "[error] run: sbatch scripts/data/prepare_omr_solution_shared2048.sh" >&2
  exit 1
fi

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
  --dataset-path "${DATASET_PATH}"
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
  --gpu-memory-utilization 0.95
  --seed 42
  --teacher-prefixes sol
  --no-save-token-metrics
)

echo "[analysis] task=${TASK} model=${MODEL_KEY} combo=${COMBO} prefix=sol backend=${BACKEND}"
echo "[analysis] dataset=${DATASET_PATH}"
echo "[analysis] output=${OUTPUT_DIR}"
echo "[analysis] score_batch=${SCORE_BATCH} gen_batch_hint=${GEN_BATCH_HINT}"

if [[ "${SKIP_GENERATE:-0}" != "1" ]]; then
  echo "[analysis] ===== phase 1: generate ====="
  python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-score
else
  echo "[analysis] ===== phase 1: skipped (SKIP_GENERATE=1; reuse rollouts.jsonl) ====="
fi

echo "[analysis] ===== phase 2: score sol ====="
python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-generate

echo "[analysis] done -> ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
