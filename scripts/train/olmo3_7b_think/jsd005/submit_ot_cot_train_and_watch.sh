#!/bin/bash
# Submit Olmo-3-7B-Think same-20650 + CoT preprocess → OPSD 100-step train,
# then a compute-node watcher that submits ckpt-100 think evals.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/olmo3_7b_think/jsd005/submit_ot_cot_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/olmo3_7b_think_ot_cot"
mkdir -p "${OUT_DIR}" log/train/olmo3-7b-think log/data
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

FILTER_SCRIPT="scripts/data/preprocess_opsd_openthoughts_olmo7bthink_st_tt_cot.sh"
TRAIN_SCRIPT="scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_cot.sh"
WATCH_SCRIPT="scripts/train/olmo3_7b_think/jsd005/sbatch_watch_ot_cot.sh"
DATASET_PATH="${ROOT}/data/openthoughts/preprocessed/openthoughts.opsd.solution.sthink_tthink.cotgold.olmo7bthink.same20650.parquet"
RUN_NAME="st_tt_clip005_1e_6_openthoughts_cot_olmo7bt"
EVAL_TAG="st_tt_ot_cot_olmo7bt_ckpt100"

chmod +x "${FILTER_SCRIPT}" "${TRAIN_SCRIPT}" "${WATCH_SCRIPT}" \
  scripts/train/olmo3_7b_think/jsd005/watch_then_eval_ot_cot.sh \
  scripts/train/olmo3_7b_think/jsd005/submit_four_think_evals_ot_cot.sh \
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
  echo -e "olmo3_7b_think\tolmo3_7b_think\t${RUN_NAME}\t${train_jid}\t${TRAIN_SCRIPT}\t${EVAL_TAG}"
} >"${MANIFEST_STAMPED}"
cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_olmo_ot_cot \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${WATCH_SCRIPT}")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
