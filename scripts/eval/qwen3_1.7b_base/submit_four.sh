#!/bin/bash
# Submit four think evals for Qwen3-1.7B-Base.
# Run from OPSD_Analysis so #SBATCH --output lands under log/eval/qwen3_1.7b_base.
#
# Usage:
#   bash scripts/eval/qwen3_1.7b_base/submit_four.sh
#   CHECKPOINT_PATH=.../checkpoint-500 EVAL_TAG=sft_think_4gpu_12k_ckpt500 \
#     bash scripts/eval/qwen3_1.7b_base/submit_four.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b-base}
EVAL_TAG=${EVAL_TAG:-$(basename "${CHECKPOINT_PATH}")}

# Job-name prefix for log/eval/qwen3_1.7b_base/<ds>/think/%x.%j.out
# Baseline: eval_1p7bb_a24 ; trained: sft_think_4gpu_12k_a24
if [[ "${EVAL_TAG}" == "qwen3-1.7b-base" || "${EVAL_TAG}" == eval_1p7bb* ]]; then
  JOB_PREFIX=eval_1p7bb
else
  JOB_PREFIX="${EVAL_TAG%_ckpt*}"
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
mkdir -p log/eval/qwen3_1.7b_base/{aime24,aime25,aime26,hmmt25}/think

echo "[submit] base_dir=${BASE_DIR}"
echo "[submit] checkpoint=${CHECKPOINT_PATH}"
echo "[submit] eval_tag=${EVAL_TAG} job_prefix=${JOB_PREFIX}"

for ds in aime24 aime25 aime26 hmmt25; do
  job_name="${JOB_PREFIX}_$(short_ds "${ds}")"
  sbatch \
    --job-name="${job_name}" \
    --output="log/eval/qwen3_1.7b_base/${ds}/think/%x.%j.out" \
    --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}" \
    "${SCRIPT_DIR}/${ds}_think.sh"
done
