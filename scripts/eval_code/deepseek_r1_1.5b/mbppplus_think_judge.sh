#!/bin/bash
#SBATCH --job-name=mb_deepseek_r1_1.5b_th_j
#SBATCH --output=log/eval_code/deepseek_r1_1.5b/mbppplus/think/%x.%j.out
#SBATCH --partition=C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#
# CPU-only EvalPlus judge for deepseek_r1_1.5b (think) on mbppplus.
# Resource / worker settings aligned with LCB *_judge.sh (C64M512G, 48c, 240G).
# Env overrides: EVALPLUS_GENERATIONS_JSON EVALPLUS_OUTPUT_JSON EVALPLUS_DATA_JSONL EVALPLUS_PASS_AT_K EVALPLUS_DATASET
#
set -euo pipefail

DATASET=mbppplus
MODEL_TAG=deepseek_r1_1.5b
DEFAULT_CKPT_BASENAME=deepseek-r1-distill-qwen-1.5b

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

EVAL_TAG=${EVAL_TAG:-${DEFAULT_CKPT_BASENAME}}
OUTPUT_JSON=${EVALPLUS_OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_think.json}
GENERATIONS_JSON=${EVALPLUS_GENERATIONS_JSON:-${OUTPUT_JSON%.json}.generations.json}
DATA_JSONL=${EVALPLUS_DATA_JSONL:-${BASE_DIR}/data/mbppplus/MbppPlus.jsonl}
PASS_AT_K=${EVALPLUS_PASS_AT_K:-1,4}
NUM_PROCESS=${NUM_PROCESS:-48}
DATASET=${EVALPLUS_DATASET:-${DATASET}}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/eval_code:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval_code/deepseek_r1_1.5b/mbppplus/think"

echo "[evalplus-judge] model_tag=${MODEL_TAG} suffix=think"
echo "[evalplus-judge] generations=${GENERATIONS_JSON}"
echo "[evalplus-judge] output=${OUTPUT_JSON}"
echo "[evalplus-judge] data=${DATA_JSONL}"
echo "[evalplus-judge] dataset=${DATASET} pass_at_k=${PASS_AT_K} workers=${NUM_PROCESS}"

python "${BASE_DIR}/eval_code/eval_evalplus_judge_local.py" \
  --generations-json "${GENERATIONS_JSON}" \
  --dataset "${DATASET}" \
  --data-jsonl "${DATA_JSONL}" \
  --output-json "${OUTPUT_JSON}" \
  --pass-at-k "${PASS_AT_K}" \
  --parallel "${NUM_PROCESS}"
