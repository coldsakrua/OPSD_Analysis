#!/bin/bash
# Submit 4 think evals for Qwen3-4B-Thinking (aime24/25/26 + hmmt25).
# Required: CHECKPOINT_PATH
# Optional: EVAL_TAG, BASE_DIR, JOB_PREFIX
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR=${BASE_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
cd "${BASE_DIR}"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH}"

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
  _run="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
  EVAL_TAG="${_run}_${_ckpt_base}"
fi

JOB_PREFIX=${JOB_PREFIX:-${EVAL_TAG}}
short_ds() {
  case "$1" in
    aime24) echo a24 ;;
    aime25) echo a25 ;;
    aime26) echo a26 ;;
    hmmt25) echo h25 ;;
    *) echo "$1" ;;
  esac
}

mkdir -p log/eval/qwen3_4b_thinking/{aime24,aime25,aime26,hmmt25}/think

echo "[submit] 4bt think ckpt=${CHECKPOINT_PATH} tag=${EVAL_TAG}"
for ds in aime24 aime25 aime26 hmmt25; do
  job_name="${JOB_PREFIX}_$(short_ds "${ds}")"
  # Slurm job-name length limit; keep short.
  job_name="${job_name:0:64}"
  sbatch \
    --job-name="${job_name}" \
    --output="log/eval/qwen3_4b_thinking/${ds}/think/%x.%j.out" \
    --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}" \
    "${SCRIPT_DIR}/${ds}_think.sh"
done
echo "[submit] done 4bt four think evals"
