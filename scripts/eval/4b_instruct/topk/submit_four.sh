#!/bin/bash
# Submit 4 nothink evals for a top-k KL 4B-Instruct checkpoint as one job array.
# Required: CHECKPOINT_PATH
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASE_DIR=${BASE_DIR:-${ROOT}}
cd "${BASE_DIR}"
: "${CHECKPOINT_PATH:?}"
SEED=${SEED:-42}

if [[ -z "${EVAL_TAG:-}" ]]; then
  _ckpt_base="$(basename "${CHECKPOINT_PATH}")"
  _job="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
  _run="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
  if [[ "${_ckpt_base}" == checkpoint-* ]]; then
    EVAL_TAG="${_run}_${_ckpt_base}"
  else
    EVAL_TAG="${_run}_${_job}_${_ckpt_base}"
  fi
fi

SCRIPT_DIR="${BASE_DIR}/scripts/eval/4b_instruct"
LOG_ROOT="log/eval/4b_instruct"
ARRAY_WORKER="${BASE_DIR}/scripts/eval/common/sbatch_four_array.sh"
OUT_JSON_FMT="${BASE_DIR}/eval_outputs/${EVAL_TAG}/__DS___4b_instruct_nothink.json"
mkdir -p "${LOG_ROOT}/array"

echo "[submit-topk-evals] 4b_instruct ckpt=${CHECKPOINT_PATH} seed=${SEED} tag=${EVAL_TAG} (array 0-3)"
jid=$(sbatch --parsable \
  --array=0-3 \
  --job-name="topk_4bi" \
  --output="${LOG_ROOT}/array/%x_%A_%a.out" \
  --time=24:00:00 \
  --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}",SEED="${SEED}",ARRAY_SCRIPT_DIR="${SCRIPT_DIR}",ARRAY_SCRIPT_SUFFIX="_nothink",ARRAY_OUTPUT_JSON_FMT="${OUT_JSON_FMT}" \
  "${ARRAY_WORKER}")
jid="${jid%%;*}"
echo "[submit-topk-evals] array_job=${jid} tasks=0-3"
