#!/bin/bash
# Submit 4 think evals (aime24/25/26 + hmmt25) as one Slurm job array.
# Required: CHECKPOINT_PATH
# Optional: SEED (default 1024), EVAL_TAG, BASE_DIR
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASE_DIR=${BASE_DIR:-${ROOT}}
cd "${BASE_DIR}"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH}"
SEED=${SEED:-1024}

if [[ -z "${EVAL_TAG:-}" ]]; then
  _ckpt_base="$(basename "${CHECKPOINT_PATH}")"
  _job="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
  _run="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
  if [[ "${_ckpt_base}" == checkpoint-* ]]; then
    EVAL_TAG="${_run}_${_ckpt_base}_seed${SEED}"
  else
    EVAL_TAG="${_run}_${_job}_${_ckpt_base}_seed${SEED}"
  fi
fi

SCRIPT_DIR="${BASE_DIR}/scripts/eval/1.7b"
LOG_ROOT="log/eval/1.7b"
ARRAY_WORKER="${BASE_DIR}/scripts/eval/common/sbatch_four_array.sh"
OUT_JSON_FMT="${BASE_DIR}/eval_outputs/${EVAL_TAG}/__DS___1.7b_think_seed${SEED}.json"
mkdir -p "${LOG_ROOT}/array"

echo "[submit-seed-evals] 1.7b think ckpt=${CHECKPOINT_PATH} seed=${SEED} tag=${EVAL_TAG} (array 0-3)"
jid=$(sbatch --parsable \
  --array=0-3 \
  --job-name="seed${SEED}_17" \
  --output="${LOG_ROOT}/array/%x_%A_%a.out" \
  --time=24:00:00 \
  --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}",SEED="${SEED}",ARRAY_SCRIPT_DIR="${SCRIPT_DIR}",ARRAY_SCRIPT_SUFFIX="_think",ARRAY_OUTPUT_JSON_FMT="${OUT_JSON_FMT}" \
  "${ARRAY_WORKER}")
jid="${jid%%;*}"
echo "[submit-seed-evals] array_job=${jid} tasks=0-3"
