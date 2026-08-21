#!/bin/bash
# Submit four nothink evals for Qwen2.5-Math-1.5B-Instruct.
# Run from OPSD_Analysis so #SBATCH --output lands under log/eval/qwen25_math_1.5b.
#
# Usage:
#   bash scripts/eval/qwen25_math_1.5b/submit_four.sh
#   CHECKPOINT_PATH=.../checkpoint-100 EVAL_TAG=st_tt_clip005_q25m15_ckpt100 \
#     bash scripts/eval/qwen25_math_1.5b/submit_four.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen2.5-math-1.5b-it}
EVAL_TAG=${EVAL_TAG:-$(basename "${CHECKPOINT_PATH}")}

# Job-name prefix for log/eval/qwen25_math_1.5b/<ds>/nothink/%x.%j.out
# Baseline: eval_q25m15_a24 ; trained: st_tt_clip005_q25m15_a24
if [[ "${EVAL_TAG}" == "qwen2.5-math-1.5b-it" || "${EVAL_TAG}" == "checkpoint-100" || "${EVAL_TAG}" == eval_q25m15* ]]; then
  JOB_PREFIX=eval_q25m15
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
if [[ ! -f "${CHECKPOINT_PATH}/model.safetensors" && ! -f "${CHECKPOINT_PATH}/model-00001-of-00001.safetensors" ]]; then
  echo "error: incomplete weights in ${CHECKPOINT_PATH}" >&2
  ls -lah "${CHECKPOINT_PATH}" >&2 || true
  exit 1
fi

cd "${BASE_DIR}"
mkdir -p log/eval/qwen25_math_1.5b/{aime24,aime25,aime26,hmmt25}/nothink

echo "[submit] base_dir=${BASE_DIR}"
echo "[submit] checkpoint=${CHECKPOINT_PATH}"
echo "[submit] eval_tag=${EVAL_TAG} job_prefix=${JOB_PREFIX}"

for ds in aime24 aime25 aime26 hmmt25; do
  job_name="${JOB_PREFIX}_$(short_ds "${ds}")"
  sbatch \
    --job-name="${job_name}" \
    --output="log/eval/qwen25_math_1.5b/${ds}/nothink/%x.%j.out" \
    --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}" \
    "${SCRIPT_DIR}/${ds}_nothink.sh"
done
