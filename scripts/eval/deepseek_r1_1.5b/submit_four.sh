#!/bin/bash
# Submit four think evals for DeepSeek-R1-Distill-Qwen-1.5B.
# Run from OPSD_Analysis so #SBATCH --output lands under log/eval/deepseek_r1_1.5b.
#
# Usage:
#   bash scripts/eval/deepseek_r1_1.5b/submit_four.sh
#   CHECKPOINT_PATH=.../checkpoint-100 EVAL_TAG=st_tt_clip005_1e6_r1_15b_ckpt100 \
#     bash scripts/eval/deepseek_r1_1.5b/submit_four.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/deepseek-r1-distill-qwen-1.5b}
EVAL_TAG=${EVAL_TAG:-$(basename "${CHECKPOINT_PATH}")}

# Job-name prefix for log/eval/deepseek_r1_1.5b/<ds>/think/%x.%j.out
# Baseline: eval_r1_15b_a24 ; trained: st_tt_clip005_1e6_r1_15b_a24
if [[ "${EVAL_TAG}" == "deepseek-r1-distill-qwen-1.5b" || "${EVAL_TAG}" == "checkpoint-100" || "${EVAL_TAG}" == eval_r1* ]]; then
  JOB_PREFIX=eval_r1_15b
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
mkdir -p log/eval/deepseek_r1_1.5b/{aime24,aime25,aime26,hmmt25}/think

echo "[submit] base_dir=${BASE_DIR}"
echo "[submit] checkpoint=${CHECKPOINT_PATH}"
echo "[submit] eval_tag=${EVAL_TAG} job_prefix=${JOB_PREFIX}"

for ds in aime24 aime25 aime26 hmmt25; do
  job_name="${JOB_PREFIX}_$(short_ds "${ds}")"
  sbatch \
    --job-name="${job_name}" \
    --output="log/eval/deepseek_r1_1.5b/${ds}/think/%x.%j.out" \
    --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}" \
    "${SCRIPT_DIR}/${ds}_think.sh"
done
