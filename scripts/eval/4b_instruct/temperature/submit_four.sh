#!/bin/bash
# Submit four nothink evals (temp=1.0) for Qwen3-4B-Instruct.
# Run from OPSD_Analysis:
#   bash scripts/eval/4b_instruct/temperature/submit_four.sh
#   CHECKPOINT_PATH=.../checkpoint-100 EVAL_TAG=snt_tnt_clip005_1e6_oti_ckpt100_t10 \
#     bash scripts/eval/4b_instruct/temperature/submit_four.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DEFAULT_BASE=/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b-instruct
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${DEFAULT_BASE}}
TEMPERATURE=${TEMPERATURE:-1.0}

if [[ -z "${EVAL_TAG:-}" ]]; then
  _ckpt_base="$(basename "${CHECKPOINT_PATH}")"
  if [[ "${_ckpt_base}" == checkpoint-* ]]; then
    _step="${_ckpt_base#checkpoint-}"
    _run="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
    EVAL_TAG="${_run}_ckpt${_step}_t10"
  elif [[ "${_ckpt_base}" == "qwen3-4b-instruct" ]]; then
    EVAL_TAG="qwen3-4b-instruct_t10"
  else
    EVAL_TAG="${_ckpt_base}_t10"
  fi
fi

if [[ -z "${JOB_PREFIX:-}" ]]; then
  if [[ "${EVAL_TAG}" == "qwen3-4b-instruct_t10" ]]; then
    JOB_PREFIX=eval_4bi_t10
  else
    JOB_PREFIX="${EVAL_TAG%_ckpt*}_t10"
    JOB_PREFIX="${JOB_PREFIX%_t10}_t10"
  fi
fi

short_ds() {
  case "$1" in
    aime24) echo a24 ;;
    aime25) echo a25 ;;
    aime26) echo a26 ;;
    hmmt25) echo h25 ;;
    *) echo "$1" ;;
  esac
}

if [[ ! -f "${CHECKPOINT_PATH}/config.json" ]]; then
  echo "error: checkpoint missing: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}/model.safetensors" && ! -f "${CHECKPOINT_PATH}/model-00001-of-00001.safetensors" && ! -f "${CHECKPOINT_PATH}/model.safetensors.index.json" ]]; then
  echo "error: incomplete weights in ${CHECKPOINT_PATH}" >&2
  ls -lah "${CHECKPOINT_PATH}" >&2 || true
  exit 1
fi

cd "${BASE_DIR}"
mkdir -p log/eval/4b_instruct/temperature/{aime24,aime25,aime26,hmmt25}/nothink

echo "[submit] base_dir=${BASE_DIR}"
echo "[submit] checkpoint=${CHECKPOINT_PATH}"
echo "[submit] temperature=${TEMPERATURE}"
echo "[submit] eval_tag=${EVAL_TAG} job_prefix=${JOB_PREFIX}"

for ds in aime24 aime25 aime26 hmmt25; do
  job_name="${JOB_PREFIX}_$(short_ds "${ds}")"
  sbatch \
    --job-name="${job_name}" \
    --output="log/eval/4b_instruct/temperature/${ds}/nothink/%x.%j.out" \
    --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}",TEMPERATURE="${TEMPERATURE}" \
    "${SCRIPT_DIR}/${ds}_nothink.sh"
done
