#!/bin/bash
# Submit 4 think evals for one GSM8K OPSD train (aime24/25/26 + hmmt25).
#
# Required:
#   MODEL_KEY=qwen3_1.7b|qwen3_4b|olmo3_7b_think
#   CHECKPOINT_PATH=.../checkpoint-100 or .../final
# Optional: EVAL_TAG, BASE_DIR, SEED
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_DIR=${BASE_DIR:-${ROOT}}
cd "${BASE_DIR}"

: "${MODEL_KEY:?}"
: "${CHECKPOINT_PATH:?}"
SEED=${SEED:-42}

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

STATE_DIR="${BASE_DIR}/log/train/gsm8k_st_tt/watch_state"
mkdir -p "${STATE_DIR}"
STATE_FILE="${STATE_DIR}/${MODEL_KEY}_${EVAL_TAG}.submitted"
if [[ -f "${STATE_FILE}" ]]; then
  echo "[skip] evals already submitted for ${MODEL_KEY}/${EVAL_TAG}"
  exit 0
fi

job_prefix="${EVAL_TAG%_checkpoint-100}"
job_prefix="${job_prefix%_ckpt100}"
job_prefix="${job_prefix:0:44}"

echo "[submit-evals] model=${MODEL_KEY} ckpt=${CHECKPOINT_PATH} tag=${EVAL_TAG} seed=${SEED}"

case "${MODEL_KEY}" in
  qwen3_1.7b|1.7b)
    SCRIPT_DIR="${BASE_DIR}/scripts/eval/1.7b"
    LOG_ROOT="log/eval/1.7b"
    mkdir -p "${LOG_ROOT}"/{aime24,aime25,aime26,hmmt25}/think
    for ds in "${DATASETS[@]}"; do
      job_name="${job_prefix}_$(short_ds "${ds}")_th"
      job_name="${job_name:0:64}"
      sbatch \
        --job-name="${job_name}" \
        --output="${LOG_ROOT}/${ds}/think/%x.%j.out" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}",SEED="${SEED}" \
        "${SCRIPT_DIR}/${ds}_think.sh"
    done
    ;;
  qwen3_4b|4b)
    SCRIPT_DIR="${BASE_DIR}/scripts/eval/4b"
    LOG_ROOT="log/eval/4b"
    mkdir -p "${LOG_ROOT}"/{aime24,aime25,aime26,hmmt25}/think
    for ds in "${DATASETS[@]}"; do
      job_name="${job_prefix}_$(short_ds "${ds}")_th"
      job_name="${job_name:0:64}"
      sbatch \
        --job-name="${job_name}" \
        --output="${LOG_ROOT}/${ds}/think/%x.%j.out" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}",SEED="${SEED}" \
        "${SCRIPT_DIR}/${ds}_think.sh"
    done
    ;;
  olmo3_7b_think|olmo_7b_think)
    SCRIPT_DIR="${BASE_DIR}/scripts/eval/olmo_7b_think"
    LOG_ROOT="log/eval/olmo_7b_think"
    mkdir -p "${LOG_ROOT}"/{aime24,aime25,aime26,hmmt25}/sgl
    for ds in "${DATASETS[@]}"; do
      job_name="${job_prefix}_$(short_ds "${ds}")_sgl"
      job_name="${job_name:0:64}"
      sbatch \
        --job-name="${job_name}" \
        --output="${LOG_ROOT}/${ds}/sgl/%x.%j.out" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}",SEED="${SEED}" \
        "${SCRIPT_DIR}/${ds}_sgl.sh"
    done
    ;;
  *)
    echo "[error] unknown MODEL_KEY=${MODEL_KEY}" >&2
    exit 1
    ;;
esac

date -Is >"${STATE_FILE}"
echo "${CHECKPOINT_PATH}" >>"${STATE_FILE}"
echo "[submit-evals] recorded ${STATE_FILE}"
