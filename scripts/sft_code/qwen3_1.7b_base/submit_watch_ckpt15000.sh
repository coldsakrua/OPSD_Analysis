#!/bin/bash
# Submit CPU watcher for 1.7B OCR SFT resume -> eval at checkpoint-15000.
#
# Defaults match the current resume:
#   train job 4071750, weights under .../sft_think_4gpu_12k_ocr/3925961
#
# Usage (from OPSD_Analysis):
#   bash scripts/sft_code/qwen3_1.7b_base/submit_watch_ckpt15000.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

TRAIN_JID=${TRAIN_JID:-4071750}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/outputs/qwen3_1.7b_base/sft_think_4gpu_12k_ocr/3925961}
CKPT_STEP=${CKPT_STEP:-15000}
MODEL_KEY=${MODEL_KEY:-qwen3_1.7b_base}
EVAL_TAG=${EVAL_TAG:-sft_think_4gpu_12k_ocr_ckpt${CKPT_STEP}}

chmod +x \
  scripts/sft_code/qwen3_1.7b_base/sbatch_watch_ckpt15000.sh \
  scripts/sft_code/qwen3_1.7b_base/watch_sft_ckpt_then_eval.sh \
  scripts/train_code/ocr/submit_three_code_evals.sh

mkdir -p log/train_code/sft/1.7b_base

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --export=ALL,BASE_DIR="${ROOT}",TRAIN_JID="${TRAIN_JID}",OUTPUT_DIR="${OUTPUT_DIR}",CKPT_STEP="${CKPT_STEP}",MODEL_KEY="${MODEL_KEY}",EVAL_TAG="${EVAL_TAG}" \
  "${ROOT}/scripts/sft_code/qwen3_1.7b_base/sbatch_watch_ckpt15000.sh")

echo "[submit-watch] train_jid=${TRAIN_JID}"
echo "[submit-watch] output_dir=${OUTPUT_DIR}"
echo "[submit-watch] ckpt_step=${CKPT_STEP} eval_tag=${EVAL_TAG}"
echo "[submit-watch] watch -> jid=${watch_jid}"
echo "[submit-watch] log: log/train_code/sft/1.7b_base/watch_ckpt15000.${watch_jid}.out"
echo "[submit-watch] evals will land under eval_outputs/${EVAL_TAG}/"
