#!/bin/bash
#SBATCH --job-name=ep_judge
#SBATCH --output=log/eval_code/%x.%j.out
#SBATCH --partition=C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#
# Generic CPU EvalPlus judge (resource settings aligned with LCB judge).
# Required: EVALPLUS_GENERATIONS_JSON EVALPLUS_OUTPUT_JSON EVALPLUS_DATASET
# Optional: EVALPLUS_DATA_JSONL EVALPLUS_PASS_AT_K NUM_PROCESS
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

: "${EVALPLUS_GENERATIONS_JSON:?}"
: "${EVALPLUS_OUTPUT_JSON:?}"
: "${EVALPLUS_DATASET:?humanevalplus|mbppplus}"
PASS_AT_K=${EVALPLUS_PASS_AT_K:-1,4}
NUM_PROCESS=${NUM_PROCESS:-48}
if [[ "${EVALPLUS_DATASET}" == "humaneval"* ]]; then
  DATA_JSONL=${EVALPLUS_DATA_JSONL:-${BASE_DIR}/data/humanevalplus/HumanEvalPlus.jsonl}
else
  DATA_JSONL=${EVALPLUS_DATA_JSONL:-${BASE_DIR}/data/mbppplus/MbppPlus.jsonl}
fi

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/eval_code:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "${EVALPLUS_OUTPUT_JSON}")" "log/eval_code"

echo "[evalplus-judge] generations=${EVALPLUS_GENERATIONS_JSON}"
echo "[evalplus-judge] output=${EVALPLUS_OUTPUT_JSON}"
echo "[evalplus-judge] dataset=${EVALPLUS_DATASET} pass_at_k=${PASS_AT_K} workers=${NUM_PROCESS}"

python "${BASE_DIR}/eval_code/eval_evalplus_judge_local.py" \
  --generations-json "${EVALPLUS_GENERATIONS_JSON}" \
  --dataset "${EVALPLUS_DATASET}" \
  --data-jsonl "${DATA_JSONL}" \
  --output-json "${EVALPLUS_OUTPUT_JSON}" \
  --pass-at-k "${PASS_AT_K}" \
  --parallel "${NUM_PROCESS}"
