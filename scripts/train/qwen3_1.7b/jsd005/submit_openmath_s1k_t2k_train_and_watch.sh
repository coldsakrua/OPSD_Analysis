#!/bin/bash
# Submit OMR s1024/t2048 filter (if needed) → Qwen3-1.7B OPSD 100-step train,
# write manifest, then start a CPU-node watcher that submits ckpt-100 think evals.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/qwen3_1.7b/jsd005/submit_openmath_s1k_t2k_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/qwen3_1p7b_omr_s1k_t2k"
mkdir -p "${OUT_DIR}" log/train/1.7b log/data
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

FILTER_SCRIPT="scripts/data/filter_opsd_openmath_qwen3_1.7b_s1024_t2048.sh"
TRAIN_SCRIPT="scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openmath_s1k_t2k.sh"
WATCH_SCRIPT="scripts/train/qwen3_1.7b/jsd005/sbatch_watch_omr_s1k_t2k.sh"
DATASET_PATH="${ROOT}/data/openmathreasoning/preprocessed/omr.opsd.solution.sthink_tthink.s1024_t2048.parquet"
RUN_NAME="st_tt_clip005_1e_6_openmath_s1k_t2k_1p7b"
EVAL_TAG="st_tt_omr_s1k_t2k_ckpt100"

chmod +x "${FILTER_SCRIPT}" "${TRAIN_SCRIPT}" "${WATCH_SCRIPT}" \
  scripts/train/qwen3_1.7b/jsd005/watch_then_eval_omr_s1k_t2k.sh \
  scripts/train/qwen3_1.7b/jsd005/submit_four_think_evals_omr_s1k_t2k.sh \
  scripts/data/filter_opsd_omr_s1024_t2048.py

filter_jid=""
if [[ -f "${DATASET_PATH}" ]]; then
  echo "[submit] dataset already exists: ${DATASET_PATH}"
else
  if [[ ! -f "${ROOT}/data/openmathreasoning/preprocessed/omr.opsd.solution.sthink_tthink.maxprompt12288.parquet" ]]; then
    echo "[error] missing 12k source parquet; cannot filter s1024/t2048" >&2
    exit 1
  fi
  filter_jid=$(sbatch --parsable --chdir="${ROOT}" "${FILTER_SCRIPT}")
  echo "[submit] filter -> jid=${filter_jid}"
fi

if [[ -n "${filter_jid}" ]]; then
  train_jid=$(sbatch --parsable --chdir="${ROOT}" --dependency="afterok:${filter_jid}" "${TRAIN_SCRIPT}")
  echo "[submit] train -> jid=${train_jid} (afterok:${filter_jid})"
else
  train_jid=$(sbatch --parsable --chdir="${ROOT}" "${TRAIN_SCRIPT}")
  echo "[submit] train -> jid=${train_jid}"
fi

{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript\teval_tag"
  echo -e "qwen3_1.7b\tqwen3_1.7b\t${RUN_NAME}\t${train_jid}\t${TRAIN_SCRIPT}\t${EVAL_TAG}"
} >"${MANIFEST_STAMPED}"
cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_1p7b_omr_s1k \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${WATCH_SCRIPT}")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
