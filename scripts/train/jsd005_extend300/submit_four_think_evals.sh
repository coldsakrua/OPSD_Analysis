#!/bin/bash
# Submit 4 think evals as one Slurm job array (aime24/25/26 + hmmt25).
#
# Required:
#   MODEL_KEY=qwen3_1.7b|olmo3_7b_think
#   CHECKPOINT_PATH=.../checkpoint-N
# Optional:
#   EVAL_TAG, SEED (default 42), BASE_DIR
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
    || [[ -f "${p}/model-00001-of-00001.safetensors" ]] \
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

if ! ckpt_ok "${CHECKPOINT_PATH}"; then
  echo "[error] incomplete checkpoint: ${CHECKPOINT_PATH}" >&2
  ls -lah "${CHECKPOINT_PATH}" >&2 || true
  exit 1
fi

step="$(basename "${CHECKPOINT_PATH}")"
step="${step#checkpoint-}"

if [[ -z "${EVAL_TAG:-}" ]]; then
  case "${MODEL_KEY}" in
    qwen3_1.7b|1.7b) EVAL_TAG="st_tt_clip005_1e6_ot_1p7b_ckpt${step}" ;;
    olmo3_7b_think|olmo_7b_think) EVAL_TAG="st_tt_clip005_1e6_olmo7bt_ckpt${step}" ;;
    *)
      _ckpt_base="$(basename "${CHECKPOINT_PATH}")"
      _run="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
      EVAL_TAG="${_run}_${_ckpt_base}"
      ;;
  esac
fi

STATE_DIR="${BASE_DIR}/log/train/jsd005_extend300/watch_state"
mkdir -p "${STATE_DIR}"
STATE_FILE="${STATE_DIR}/${MODEL_KEY}_${EVAL_TAG}.submitted"
if [[ -f "${STATE_FILE}" ]]; then
  echo "[skip] evals already submitted for ${MODEL_KEY}/${EVAL_TAG} ($(cat "${STATE_FILE}"))"
  exit 0
fi

echo "[submit-evals] model=${MODEL_KEY} ckpt=${CHECKPOINT_PATH} seed=${SEED} tag=${EVAL_TAG}"

case "${MODEL_KEY}" in
  qwen3_1.7b|1.7b)
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" EVAL_TAG="${EVAL_TAG}" SEED="${SEED}" BASE_DIR="${BASE_DIR}" \
      bash "${BASE_DIR}/scripts/eval/1.7b/seed/submit_four_seed.sh"
    ;;
  olmo3_7b_think|olmo_7b_think)
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" EVAL_TAG="${EVAL_TAG}" SEED="${SEED}" BASE_DIR="${BASE_DIR}" \
      bash "${BASE_DIR}/scripts/eval/olmo_7b_think/seed/submit_four_seed.sh"
    ;;
  *)
    echo "[error] unknown MODEL_KEY=${MODEL_KEY}" >&2
    exit 1
    ;;
esac

date -Is >"${STATE_FILE}"
echo "${CHECKPOINT_PATH}" >>"${STATE_FILE}"
echo "[submit-evals] recorded ${STATE_FILE}"
