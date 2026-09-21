#!/bin/bash
# Submit OMR postthink filter → Qwen3-1.7B OPSD 1k + 4k trains, then a CPU watcher
# that submits ckpt-100 think evals for both.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/qwen3_1.7b/jsd005/submit_openmath_postthink_1k_4k_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/qwen3_1p7b_omr_postthink"
mkdir -p "${OUT_DIR}" log/train/1.7b log/data
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

FILTER_SCRIPT="scripts/data/filter_opsd_openmath_qwen3_1.7b_postthink_1k_4k.sh"
TRAIN_1K="scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openmath_postthink_t1k.sh"
TRAIN_4K="scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openmath_postthink_t4k.sh"
WATCH_SCRIPT="scripts/train/qwen3_1.7b/jsd005/sbatch_watch_omr_postthink.sh"
DS_1K="${ROOT}/data/openmathreasoning/preprocessed/omr.opsd.solution.postthink.sthink_tthink.maxprompt1024.parquet"
DS_4K="${ROOT}/data/openmathreasoning/preprocessed/omr.opsd.solution.postthink.sthink_tthink.t2048_4096.parquet"
SRC_12K="${ROOT}/data/openmathreasoning/preprocessed/omr.opsd.solution.sthink_tthink.maxprompt12288.parquet"
RUN_1K="st_tt_clip005_1e_6_openmath_postthink_t1k_1p7b"
RUN_4K="st_tt_clip005_1e_6_openmath_postthink_t4k_1p7b"
EVAL_1K="st_tt_omr_postthink_t1k_ckpt100"
EVAL_4K="st_tt_omr_postthink_t4k_ckpt100"

chmod +x "${FILTER_SCRIPT}" "${TRAIN_1K}" "${TRAIN_4K}" "${WATCH_SCRIPT}" \
  scripts/train/qwen3_1.7b/jsd005/watch_then_eval_omr_postthink.sh \
  scripts/train/qwen3_1.7b/jsd005/submit_four_think_evals_omr_postthink.sh \
  scripts/data/filter_opsd_omr_postthink.py

filter_jid=""
if [[ -f "${DS_1K}" && -f "${DS_4K}" ]]; then
  echo "[submit] both postthink parquets already exist"
else
  if [[ ! -f "${SRC_12K}" ]]; then
    echo "[error] missing 12k source parquet: ${SRC_12K}" >&2
    exit 1
  fi
  filter_jid=$(sbatch --parsable --chdir="${ROOT}" "${FILTER_SCRIPT}")
  echo "[submit] filter -> jid=${filter_jid}"
fi

if [[ -n "${filter_jid}" ]]; then
  train_1k_jid=$(sbatch --parsable --chdir="${ROOT}" --dependency="afterok:${filter_jid}" "${TRAIN_1K}")
  train_4k_jid=$(sbatch --parsable --chdir="${ROOT}" --dependency="afterok:${filter_jid}" "${TRAIN_4K}")
  echo "[submit] train 1k -> jid=${train_1k_jid} (afterok:${filter_jid})"
  echo "[submit] train 4k -> jid=${train_4k_jid} (afterok:${filter_jid})"
else
  train_1k_jid=$(sbatch --parsable --chdir="${ROOT}" "${TRAIN_1K}")
  train_4k_jid=$(sbatch --parsable --chdir="${ROOT}" "${TRAIN_4K}")
  echo "[submit] train 1k -> jid=${train_1k_jid}"
  echo "[submit] train 4k -> jid=${train_4k_jid}"
fi

{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript\teval_tag"
  echo -e "qwen3_1.7b\tqwen3_1.7b\t${RUN_1K}\t${train_1k_jid}\t${TRAIN_1K}\t${EVAL_1K}"
  echo -e "qwen3_1.7b\tqwen3_1.7b\t${RUN_4K}\t${train_4k_jid}\t${TRAIN_4K}\t${EVAL_4K}"
} >"${MANIFEST_STAMPED}"
cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_1p7b_omr_pt \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${WATCH_SCRIPT}")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
