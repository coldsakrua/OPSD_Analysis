#!/bin/bash
#SBATCH --job-name=he_4bb_th_j
#SBATCH --output=log/eval_code/4b_base/humanevalplus/think/%x.%j.out
#SBATCH --partition=C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#
# CPU-only judge for 4b_base (think) on humanevalplus.
set -euo pipefail

DATASET=humanevalplus
MODEL_TAG=4b_base
DEFAULT_CKPT_BASENAME=qwen3-4b-base

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
DATA_JSONL=${EVALPLUS_DATA_JSONL:-${BASE_DIR}/data/humanevalplus/HumanEvalPlus.jsonl}
PASS_AT_K=${EVALPLUS_PASS_AT_K:-1,4}
NUM_PROCESS=${NUM_PROCESS:-48}
DATASET=${EVALPLUS_DATASET:-${DATASET}}
# sft_think rename-extract: SFT paths only (never OCR / base / other tags).
EXTRACT_PROFILE=${EXTRACT_PROFILE:-${EVALPLUS_EXTRACT_PROFILE:-auto}}
if [[ "${EXTRACT_PROFILE}" == "auto" ]]; then
  _blob="$(echo "${EVAL_TAG:-} ${GENERATIONS_JSON} ${OUTPUT_JSON}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${_blob}" == *sft_think* || "${_blob}" == */sft/* || "${_blob}" == *[_/.-]sft[_/.-]* ]]; then
    EXTRACT_PROFILE=sft_think
  fi
fi

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/eval_code:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval_code/4b_base/humanevalplus/think"

echo "[evalplus-judge] model_tag=${MODEL_TAG} suffix=think"
echo "[evalplus-judge] generations=${GENERATIONS_JSON}"
echo "[evalplus-judge] output=${OUTPUT_JSON}"
echo "[evalplus-judge] data=${DATA_JSONL}"
echo "[evalplus-judge] pass_at_k=${PASS_AT_K} workers=${NUM_PROCESS}"
echo "[evalplus-judge] extract_profile=${EXTRACT_PROFILE}"

python "${BASE_DIR}/eval_code/eval_evalplus_judge_local.py" \
  --generations-json "${GENERATIONS_JSON}" \
  --dataset "${DATASET}" \
  --data-jsonl "${DATA_JSONL}" \
  --output-json "${OUTPUT_JSON}" \
  --pass-at-k "${PASS_AT_K}" \
  --parallel "${NUM_PROCESS}" \
  --extract-profile "${EXTRACT_PROFILE}"
