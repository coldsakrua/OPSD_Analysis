#!/bin/bash
# Submit OT same-21981 + CoT preprocess (compute node) → Qwen3-1.7B OPSD
# 100-step train, write manifest, then start a compute-node watcher that
# submits ckpt-100 think evals (aime24/25/26 + hmmt25).
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/qwen3_1.7b/jsd005/submit_ot_cot_t4096_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/qwen3_1p7b_ot_cot_t4096"
mkdir -p "${OUT_DIR}" log/train/1.7b log/data
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

FILTER_SCRIPT="scripts/data/preprocess_opsd_openthoughts_qwen3_1.7b_st_tt_cot_t4096.sh"
TRAIN_SCRIPT="scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_cot_t4096.sh"
WATCH_SCRIPT="scripts/train/qwen3_1.7b/jsd005/sbatch_watch_ot_cot_t4096.sh"
DATASET_PATH="${ROOT}/data/openthoughts/preprocessed/openthoughts.opsd.solution.sthink_tthink.cotgold.same21981.parquet"
RUN_NAME="st_tt_clip005_1e_6_openthoughts_cot_t4096_1p7b"
EVAL_TAG="st_tt_ot_cot_t4096_ckpt100"

chmod +x "${FILTER_SCRIPT}" "${TRAIN_SCRIPT}" "${WATCH_SCRIPT}" \
  scripts/train/qwen3_1.7b/jsd005/watch_then_eval_ot_cot_t4096.sh \
  scripts/train/qwen3_1.7b/jsd005/submit_four_think_evals_ot_cot_t4096.sh \
  scripts/data/preprocess_opsd_openthoughts_cot_solution.py

filter_jid=""
if [[ -f "${DATASET_PATH}" ]]; then
  echo "[submit] dataset already exists: ${DATASET_PATH}"
else
  filter_jid=$(sbatch --parsable --chdir="${ROOT}" "${FILTER_SCRIPT}")
  echo "[submit] preprocess -> jid=${filter_jid}"
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
  --job-name=watch_1p7b_ot_cot \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${WATCH_SCRIPT}")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
