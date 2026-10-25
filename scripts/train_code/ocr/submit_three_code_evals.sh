#!/bin/bash
# Submit 3 code evals for one OCR OPSD train: GPU generate first, then CPU judge.
# Generate scripts already auto-submit judge after *.generations.json is written.
# Do NOT sbatch *_judge.sh here.
#
# Required: MODEL_KEY CHECKPOINT_PATH
# Optional: EVAL_TAG BASE_DIR SEED
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_DIR=${BASE_DIR:-${ROOT}}
cd "${BASE_DIR}"

: "${MODEL_KEY:?}"
: "${CHECKPOINT_PATH:?}"
SEED=${SEED:-42}
AUTO_JUDGE=${AUTO_JUDGE:-1}

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

STATE_DIR="${BASE_DIR}/log/train_code/ocr/watch_state"
mkdir -p "${STATE_DIR}"
STATE_FILE="${STATE_DIR}/${MODEL_KEY}_${EVAL_TAG}.submitted"
if [[ -f "${STATE_FILE}" ]]; then
  echo "[skip] evals already submitted for ${MODEL_KEY}/${EVAL_TAG}"
  exit 0
fi

job_prefix="${EVAL_TAG%_checkpoint-100}"
job_prefix="${job_prefix%_ckpt100}"
job_prefix="${job_prefix:0:44}"

short_ds() {
  case "$1" in
    lcb_v6) echo lcb ;;
    humanevalplus) echo he ;;
    mbppplus) echo mb ;;
    *) echo "$1" ;;
  esac
}

submit_one() {
  local ds=$1 script=$2 log_out=$3 suffix=$4
  local job_name="${job_prefix}_$(short_ds "${ds}")_${suffix}"
  job_name="${job_name:0:64}"
  mkdir -p "$(dirname "${log_out}")"
  echo "[submit-evals] generate ${ds} script=${script}"
  sbatch \
    --job-name="${job_name}" \
    --output="${log_out}" \
    --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}",SEED="${SEED}",AUTO_JUDGE="${AUTO_JUDGE}" \
    --chdir="${BASE_DIR}" \
    "${script}"
}

echo "[submit-evals] model=${MODEL_KEY} ckpt=${CHECKPOINT_PATH} tag=${EVAL_TAG} seed=${SEED} auto_judge=${AUTO_JUDGE}"

case "${MODEL_KEY}" in
  qwen3_1.7b|1.7b)
    DIR="${BASE_DIR}/scripts/eval_code/1.7b"
    LOG="log/eval_code/1.7b"
    submit_one lcb_v6 "${DIR}/lcb_v6_think.sh" "${LOG}/lcb_v6/think/%x.%j.out" th
    submit_one humanevalplus "${DIR}/humanevalplus_think.sh" "${LOG}/humanevalplus/think/%x.%j.out" th
    submit_one mbppplus "${DIR}/mbppplus_think.sh" "${LOG}/mbppplus/think/%x.%j.out" th
    ;;
  qwen3_1.7b_base|1.7b_base)
    DIR="${BASE_DIR}/scripts/eval_code/1.7b_base"
    LOG="log/eval_code/1.7b_base"
    submit_one lcb_v6 "${DIR}/lcb_v6_think.sh" "${LOG}/lcb_v6/think/%x.%j.out" th
    submit_one humanevalplus "${DIR}/humanevalplus_think.sh" "${LOG}/humanevalplus/think/%x.%j.out" th
    submit_one mbppplus "${DIR}/mbppplus_think.sh" "${LOG}/mbppplus/think/%x.%j.out" th
    ;;
  qwen3_4b|4b)
    DIR="${BASE_DIR}/scripts/eval_code/4b"
    LOG="log/eval_code/4b"
    submit_one lcb_v6 "${DIR}/lcb_v6_think.sh" "${LOG}/lcb_v6/think/%x.%j.out" th
    submit_one humanevalplus "${DIR}/humanevalplus_think.sh" "${LOG}/humanevalplus/think/%x.%j.out" th
    submit_one mbppplus "${DIR}/mbppplus_think.sh" "${LOG}/mbppplus/think/%x.%j.out" th
    ;;
  qwen3_4b_base|4b_base)
    DIR="${BASE_DIR}/scripts/eval_code/4b_base"
    LOG="log/eval_code/4b_base"
    submit_one lcb_v6 "${DIR}/lcb_v6_think.sh" "${LOG}/lcb_v6/think/%x.%j.out" th
    submit_one humanevalplus "${DIR}/humanevalplus_think.sh" "${LOG}/humanevalplus/think/%x.%j.out" th
    submit_one mbppplus "${DIR}/mbppplus_think.sh" "${LOG}/mbppplus/think/%x.%j.out" th
    ;;
  qwen3_8b|8b)
    DIR="${BASE_DIR}/scripts/eval_code/8b"
    LOG="log/eval_code/8b"
    submit_one lcb_v6 "${DIR}/lcb_v6_think.sh" "${LOG}/lcb_v6/think/%x.%j.out" th
    submit_one humanevalplus "${DIR}/humanevalplus_think.sh" "${LOG}/humanevalplus/think/%x.%j.out" th
    submit_one mbppplus "${DIR}/mbppplus_think.sh" "${LOG}/mbppplus/think/%x.%j.out" th
    ;;
  qwen3_4b_instruct|4b_instruct)
    DIR="${BASE_DIR}/scripts/eval_code/4b_instruct"
    LOG="log/eval_code/4b_instruct"
    submit_one lcb_v6 "${DIR}/lcb_v6_nothink.sh" "${LOG}/lcb_v6/nothink/%x.%j.out" nt
    submit_one humanevalplus "${DIR}/humanevalplus_nothink.sh" "${LOG}/humanevalplus/nothink/%x.%j.out" nt
    submit_one mbppplus "${DIR}/mbppplus_nothink.sh" "${LOG}/mbppplus/nothink/%x.%j.out" nt
    ;;
  olmo3_7b_instruct|olmo3_7b_it)
    DIR="${BASE_DIR}/scripts/eval_code/olmo3-7b-it"
    LOG="log/eval_code/olmo3-7b-it"
    submit_one lcb_v6 "${DIR}/lcb_v6_nothink.sh" "${LOG}/lcb_v6/sgl/%x.%j.out" nt
    submit_one humanevalplus "${DIR}/humanevalplus_sgl_nothink.sh" "${LOG}/humanevalplus/sgl/%x.%j.out" nt
    submit_one mbppplus "${DIR}/mbppplus_sgl_nothink.sh" "${LOG}/mbppplus/sgl/%x.%j.out" nt
    ;;
  olmo3_7b_think)
    DIR="${BASE_DIR}/scripts/eval_code/olmo3-7b-think"
    LOG="log/eval_code/olmo3-7b-think"
    submit_one lcb_v6 "${DIR}/lcb_v6_think.sh" "${LOG}/lcb_v6/sgl/%x.%j.out" th
    submit_one humanevalplus "${DIR}/humanevalplus_sgl_think.sh" "${LOG}/humanevalplus/sgl/%x.%j.out" th
    submit_one mbppplus "${DIR}/mbppplus_sgl_think.sh" "${LOG}/mbppplus/sgl/%x.%j.out" th
    ;;
  *)
    echo "[error] unknown MODEL_KEY=${MODEL_KEY}" >&2
    exit 1
    ;;
esac

date -Is >"${STATE_FILE}"
echo "${CHECKPOINT_PATH}" >>"${STATE_FILE}"
echo "[submit-evals] recorded ${STATE_FILE} (generate jobs will auto-submit CPU judges)"
