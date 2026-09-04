#!/bin/bash
#SBATCH --job-name=da26_st_tt_8b_easy
#SBATCH --output=log/data_analysis/26/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
set -euo pipefail

# 2.6 cotlen-easy st_tt on qwen3_8b
# OT cotlen band=easy: prompt=2048 completion=38912; n=4 rollouts.
# Default: generate + boxed accuracy only. Preference/data-analysis: RUN_PREFERENCE=1 (or submit_preference.sh).
# Rollout: temp=1.1 top_p=0.95 top_k=20 max_prompt=2048 max_completion=38912

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/scripts/data_analysis/outputs/cotlen/qwen3_8b/st_tt_easy_${JOB_TAG}}

TASK="cotlen"
MODEL_KEY="qwen3_8b"
COMBO="st_tt"
MODEL_PATH="/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-8b"
CONDA_ENV="anchor"
BACKEND="vllm"
NUM_PROMPTS=${NUM_PROMPTS:-2048}
N_ROLLOUTS=${N_ROLLOUTS:-4}
MAX_PROMPT=${MAX_PROMPT:-2048}
MAX_COMPLETION=${MAX_COMPLETION:-38912}
SCORE_BATCH=${SCORE_BATCH:-1}
GEN_BATCH_HINT=${GEN_BATCH_HINT:-2}

mkdir -p "${OUTPUT_DIR}" "${BASE_DIR}/log/data_analysis/26"

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
EXTRA_ARGS+=(--cotlen-band "easy")

echo "[analysis] task=${TASK} model=${MODEL_KEY} combo=${COMBO} backend=${BACKEND}"
echo "[analysis] output=${OUTPUT_DIR}"

if [[ "${SKIP_GENERATE:-0}" != "1" ]]; then
  echo "[analysis] ===== phase A: generate + accuracy ====="
  python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-score
elif [[ "${RUN_ACCURACY:-1}" == "1" ]]; then
  echo "[analysis] ===== phase A: accuracy only (reuse rollouts.jsonl) ====="
  python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-generate --skip-score
else
  echo "[analysis] ===== phase A: skipped ====="
fi

if [[ "${RUN_PREFERENCE:-0}" == "1" ]]; then
  echo "[analysis] ===== phase B: preference / data analysis ====="
  python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-generate --skip-accuracy
else
  echo "[analysis] ===== phase B: preference skipped (set RUN_PREFERENCE=1) ====="
fi

echo "[analysis] done -> ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
