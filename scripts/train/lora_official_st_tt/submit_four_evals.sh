#!/bin/bash
# Submit 4 think evals (aime24/25/26 + hmmt25) for one official-OPSD LoRA checkpoint-100.
#
# Required:
#   MODEL_KEY=qwen3_4b_thinking|olmo3_7b_think
#   CHECKPOINT_PATH=.../checkpoint-100   (PEFT adapter dir)
# Optional: EVAL_TAG, BASE_MODEL_PATH, MERGED_PATH (olmo only)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

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

if [[ -z "${EVAL_TAG:-}" ]]; then
  _run="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
  EVAL_TAG="${_run}_ckpt100"
fi
if [[ "${EVAL_TAG}" != *lora* ]]; then
  EVAL_TAG="lora_${EVAL_TAG}"
fi

adapter_ok() {
  [[ -f "$1/adapter_config.json" ]] && {
    [[ -f "$1/adapter_model.safetensors" ]] || [[ -f "$1/adapter_model.bin" ]]
  }
}

if ! adapter_ok "${CHECKPOINT_PATH}"; then
  echo "[error] not a LoRA adapter dir: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ "$(basename "${CHECKPOINT_PATH}")" != "checkpoint-100" ]]; then
  echo "[error] expected checkpoint-100, got: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

MODEL_ROOT=/gpfs/share/home/2501210611/labShare/2501210611/model
STATE_DIR="${ROOT}/log/train/lora_official_st_tt/watch_state"
mkdir -p "${STATE_DIR}"
STATE_FILE="${STATE_DIR}/${MODEL_KEY}_${EVAL_TAG}.submitted"

if [[ -f "${STATE_FILE}" ]]; then
  echo "[skip] evals already submitted for ${MODEL_KEY}/${EVAL_TAG} ($(cat "${STATE_FILE}"))"
  exit 0
fi

echo "[submit-evals] model=${MODEL_KEY} ckpt=${CHECKPOINT_PATH}"
echo "[submit-evals] eval_tag=${EVAL_TAG}"

case "${MODEL_KEY}" in
  qwen3_4b_thinking)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-${MODEL_ROOT}/qwen3-4b-thinking}
    SCRIPT_DIR="${ROOT}/scripts/eval/qwen3_4b_thinking/lora"
    LOG_ROOT="log/eval/qwen3_4b_thinking/lora"
    mkdir -p "${LOG_ROOT}"/{aime24,aime25,aime26,hmmt25}/think
    for ds in "${DATASETS[@]}"; do
      sbatch \
        --job-name="lora4bt_$(short_ds "${ds}")_c100" \
        --output="${LOG_ROOT}/${ds}/think/%x.%j.out" \
        --export=ALL,BASE_DIR="${ROOT}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",BASE_MODEL_PATH="${BASE_MODEL_PATH}",EVAL_TAG="${EVAL_TAG}" \
        "${SCRIPT_DIR}/${ds}_think.sh"
    done
    ;;
  olmo3_7b_think)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-${MODEL_ROOT}/olmo-3-7b-think}
    MERGED_PATH=${MERGED_PATH:-$(dirname "${CHECKPOINT_PATH}")/merged_ckpt100}
    MERGE_SCRIPT="${ROOT}/scripts/train/beta_opsd/merge_olmo_lora.sh"
    mkdir -p "$(dirname "${MERGED_PATH}")" log/eval/olmo_7b_think/lora/merge \
      log/eval/olmo_7b_think/{aime24,aime25,aime26,hmmt25}/sgl
    merge_jid=$(sbatch --parsable \
      --job-name="lora_olmo_merge_c100" \
      --output="log/eval/olmo_7b_think/lora/merge/%x.%j.out" \
      --export=ALL,BASE_DIR="${ROOT}",BASE_MODEL_PATH="${BASE_MODEL_PATH}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",MERGED_PATH="${MERGED_PATH}",EVAL_TAG="${EVAL_TAG}" \
      "${MERGE_SCRIPT}")
    echo "[submit-evals] olmo-think merge job=${merge_jid} -> ${MERGED_PATH}"
    for ds in "${DATASETS[@]}"; do
      sbatch \
        --dependency=afterok:"${merge_jid}" \
        --job-name="lora_olmot_$(short_ds "${ds}")_c100" \
        --output="log/eval/olmo_7b_think/${ds}/sgl/%x.%j.out" \
        --export=ALL,BASE_DIR="${ROOT}",CHECKPOINT_PATH="${MERGED_PATH}",EVAL_TAG="${EVAL_TAG}" \
        "${ROOT}/scripts/eval/olmo_7b_think/${ds}_sgl.sh"
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
