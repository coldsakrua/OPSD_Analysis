#!/bin/bash
# Route finished robustness train → 4 evals with default SEED=42.
# Required: MODEL_KEY, CHECKPOINT_PATH
# Optional: EVAL_TAG, BASE_DIR
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_DIR=${BASE_DIR:-${ROOT}}
cd "${BASE_DIR}"

: "${MODEL_KEY:?}"
: "${CHECKPOINT_PATH:?}"
SEED=${SEED:-42}

ckpt_ok() {
  local p=$1
  [[ -f "${p}/config.json" ]] || return 1
  [[ -f "${p}/model.safetensors" ]] \
    || [[ -f "${p}/model.safetensors.index.json" ]] \
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

if ! ckpt_ok "${CHECKPOINT_PATH}"; then
  echo "[error] incomplete checkpoint: ${CHECKPOINT_PATH}" >&2
  ls -lah "${CHECKPOINT_PATH}" >&2 || true
  exit 1
fi

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

echo "[submit-evals] model=${MODEL_KEY} ckpt=${CHECKPOINT_PATH} seed=${SEED} tag=${EVAL_TAG}"

case "${MODEL_KEY}" in
  qwen3_1.7b|1.7b)
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" EVAL_TAG="${EVAL_TAG}" SEED="${SEED}" BASE_DIR="${BASE_DIR}" \
      bash "${BASE_DIR}/scripts/eval/1.7b/topk/submit_four.sh"
    ;;
  olmo3_7b_think|olmo_7b_think)
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" EVAL_TAG="${EVAL_TAG}" SEED="${SEED}" BASE_DIR="${BASE_DIR}" \
      bash "${BASE_DIR}/scripts/eval/olmo_7b_think/topk/submit_four.sh"
    ;;
  qwen3_4b_instruct|4b_instruct)
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" EVAL_TAG="${EVAL_TAG}" SEED="${SEED}" BASE_DIR="${BASE_DIR}" \
      bash "${BASE_DIR}/scripts/eval/4b_instruct/topk/submit_four.sh"
    ;;
  *)
    echo "[error] unknown MODEL_KEY=${MODEL_KEY}" >&2
    exit 1
    ;;
esac
