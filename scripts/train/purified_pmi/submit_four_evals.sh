#!/bin/bash
# Submit 4 think evals (aime24/25/26 + hmmt25) for one Purified-OPSD LoRA final adapter.
#
# Required:
#   MODEL_KEY=qwen3_1.7b|qwen3_4b|qwen3_4b_thinking|olmo3_7b_think
#   CHECKPOINT_PATH=.../final   (PEFT adapter dir)
# Optional:
#   EVAL_TAG, BASE_MODEL_PATH, MERGED_PATH (olmo only)
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
  _job="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
  EVAL_TAG="pmi_${_run}_${_job}_final"
fi
if [[ "${EVAL_TAG}" != *lora* && "${EVAL_TAG}" != *pmi* ]]; then
  EVAL_TAG="pmi_lora_${EVAL_TAG}"
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

MODEL_ROOT=/gpfs/share/home/2501210611/labShare/2501210611/model
STATE_DIR="${ROOT}/log/train/purified_pmi/watch_state"
mkdir -p "${STATE_DIR}" log/eval/purified_pmi
STATE_FILE="${STATE_DIR}/${EVAL_TAG}.submitted"

if [[ -f "${STATE_FILE}" ]]; then
  echo "[skip] evals already submitted for ${EVAL_TAG} ($(cat "${STATE_FILE}"))"
  exit 0
fi

echo "[submit-evals] model=${MODEL_KEY} ckpt=${CHECKPOINT_PATH}"
echo "[submit-evals] eval_tag=${EVAL_TAG}"

case "${MODEL_KEY}" in
  qwen3_1.7b)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-${MODEL_ROOT}/qwen3-1.7b}
    SCRIPT_DIR="${ROOT}/scripts/eval/1.7b/lora"
    LOG_ROOT="log/eval/1.7b/lora"
    for ds in "${DATASETS[@]}"; do
      sbatch \
        --job-name="pmi17_$(short_ds "${ds}")_${EVAL_TAG: -16}" \
        --output="${LOG_ROOT}/${ds}/think/%x.%j.out" \
        --export=ALL,BASE_DIR="${ROOT}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",BASE_MODEL_PATH="${BASE_MODEL_PATH}",EVAL_TAG="${EVAL_TAG}" \
        "${SCRIPT_DIR}/${ds}_think.sh"
    done
    ;;
  qwen3_4b)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-${MODEL_ROOT}/qwen3-4b}
    SCRIPT_DIR="${ROOT}/scripts/eval/4b/lora"
    LOG_ROOT="log/eval/4b/lora"
    for ds in "${DATASETS[@]}"; do
      sbatch \
        --job-name="pmi4b_$(short_ds "${ds}")_${EVAL_TAG: -16}" \
        --output="${LOG_ROOT}/${ds}/think/%x.%j.out" \
        --export=ALL,BASE_DIR="${ROOT}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",BASE_MODEL_PATH="${BASE_MODEL_PATH}",EVAL_TAG="${EVAL_TAG}" \
        "${SCRIPT_DIR}/${ds}_think.sh"
    done
    ;;
  qwen3_4b_thinking)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-${MODEL_ROOT}/qwen3-4b-thinking}
    SCRIPT_DIR="${ROOT}/scripts/eval/qwen3_4b_thinking/lora"
    LOG_ROOT="log/eval/qwen3_4b_thinking/lora"
    for ds in "${DATASETS[@]}"; do
      sbatch \
        --job-name="pmi4bt_$(short_ds "${ds}")_${EVAL_TAG: -14}" \
        --output="${LOG_ROOT}/${ds}/think/%x.%j.out" \
        --export=ALL,BASE_DIR="${ROOT}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",BASE_MODEL_PATH="${BASE_MODEL_PATH}",EVAL_TAG="${EVAL_TAG}" \
        "${SCRIPT_DIR}/${ds}_think.sh"
    done
    ;;
  olmo3_7b_think)
    BASE_MODEL_PATH=${BASE_MODEL_PATH:-${MODEL_ROOT}/olmo-3-7b-think}
    MERGED_PATH=${MERGED_PATH:-$(dirname "${CHECKPOINT_PATH}")/merged_final}
    # Merge LoRA → dense (CPU/GPU job), then chain 4 SGLang evals.
    MERGE_SCRIPT="${ROOT}/scripts/train/purified_pmi/merge_olmo_lora.sh"
    mkdir -p "$(dirname "${MERGED_PATH}")" log/eval/olmo_7b_think/lora/merge
    merge_jid=$(sbatch --parsable \
      --job-name="pmi_olmo_merge_${EVAL_TAG: -12}" \
      --output="log/eval/olmo_7b_think/lora/merge/%x.%j.out" \
      --export=ALL,BASE_DIR="${ROOT}",BASE_MODEL_PATH="${BASE_MODEL_PATH}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",MERGED_PATH="${MERGED_PATH}",EVAL_TAG="${EVAL_TAG}" \
      "${MERGE_SCRIPT}")
    echo "[submit-evals] olmo merge job=${merge_jid} -> ${MERGED_PATH}"
    for ds in "${DATASETS[@]}"; do
      sbatch \
        --dependency=afterok:"${merge_jid}" \
        --job-name="pmi_olmo_$(short_ds "${ds}")_${EVAL_TAG: -12}" \
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
echo "[submit-evals] recorded ${STATE_FILE}"
