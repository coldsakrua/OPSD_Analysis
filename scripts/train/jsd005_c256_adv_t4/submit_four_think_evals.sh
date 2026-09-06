#!/bin/bash
# Submit 4 think evals (aime24/25/26 + hmmt25) for one finished c256+adv_t4 train.
#
# Required:
#   MODEL_KEY=qwen3_1.7b|qwen3_4b|qwen3_4b_thinking|olmo3_7b_think|mimo_7b_rl
#   CHECKPOINT_PATH=.../checkpoint-100 or .../final
# Optional:
#   EVAL_TAG, BASE_DIR
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_DIR=${BASE_DIR:-${ROOT}}
cd "${BASE_DIR}"

: "${MODEL_KEY:?}"
: "${CHECKPOINT_PATH:?}"

DATASETS=(aime24 aime25 aime26 hmmt25)
short_ds() {
  case "$1" in
    aime24) echo a24 ;;
    aime25) echo a25 ;;
    aime26) echo a26 ;;
    hmmt25) echo h25 ;;
    *) echo "$1" ;;
  esac
}

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

STATE_DIR="${BASE_DIR}/log/train/jsd005_c256_adv_t4/watch_state"
mkdir -p "${STATE_DIR}"
STATE_FILE="${STATE_DIR}/${MODEL_KEY}_${EVAL_TAG}.submitted"
if [[ -f "${STATE_FILE}" ]]; then
  echo "[skip] evals already submitted for ${MODEL_KEY}/${EVAL_TAG} ($(cat "${STATE_FILE}"))"
  exit 0
fi

echo "[submit-evals] model=${MODEL_KEY} ckpt=${CHECKPOINT_PATH}"
echo "[submit-evals] eval_tag=${EVAL_TAG}"

case "${MODEL_KEY}" in
  qwen3_1.7b)
    SCRIPT_DIR="${BASE_DIR}/scripts/eval/1.7b"
    LOG_ROOT="log/eval/1.7b"
    PREFIX="advt4_17"
    for ds in "${DATASETS[@]}"; do
      sbatch \
        --job-name="${PREFIX}_$(short_ds "${ds}")" \
        --output="${LOG_ROOT}/${ds}/think/%x.%j.out" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}" \
        "${SCRIPT_DIR}/${ds}_think.sh"
    done
    ;;
  qwen3_4b)
    SCRIPT_DIR="${BASE_DIR}/scripts/eval/4b"
    LOG_ROOT="log/eval/4b"
    PREFIX="advt4_4b"
    for ds in "${DATASETS[@]}"; do
      sbatch \
        --job-name="${PREFIX}_$(short_ds "${ds}")" \
        --output="${LOG_ROOT}/${ds}/think/%x.%j.out" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}" \
        "${SCRIPT_DIR}/${ds}_think.sh"
    done
    ;;
  qwen3_4b_thinking)
    SCRIPT_DIR="${BASE_DIR}/scripts/eval/qwen3_4b_thinking"
    LOG_ROOT="log/eval/qwen3_4b_thinking"
    PREFIX="advt4_4bt"
    for ds in "${DATASETS[@]}"; do
      sbatch \
        --job-name="${PREFIX}_$(short_ds "${ds}")" \
        --output="${LOG_ROOT}/${ds}/think/%x.%j.out" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}" \
        "${SCRIPT_DIR}/${ds}_think.sh"
    done
    ;;
  olmo3_7b_think)
    SCRIPT_DIR="${BASE_DIR}/scripts/eval/olmo_7b_think"
    LOG_ROOT="log/eval/olmo_7b_think"
    PREFIX="advt4_olmo"
    for ds in "${DATASETS[@]}"; do
      sbatch \
        --job-name="${PREFIX}_$(short_ds "${ds}")" \
        --output="${LOG_ROOT}/${ds}/sgl/%x.%j.out" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}" \
        "${SCRIPT_DIR}/${ds}_sgl.sh"
    done
    ;;
  mimo_7b_rl)
    # MiMo freeform CoT via SGLang (--no-thinking flag; template has no enable_thinking).
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" EVAL_TAG="${EVAL_TAG}" \
      bash "${BASE_DIR}/scripts/eval/mimo_7b_rl/submit_four.sh"
    ;;
  *)
    echo "[error] unknown MODEL_KEY=${MODEL_KEY}" >&2
    exit 1
    ;;
esac

date -Is >"${STATE_FILE}"
echo "[submit-evals] recorded ${STATE_FILE}"
