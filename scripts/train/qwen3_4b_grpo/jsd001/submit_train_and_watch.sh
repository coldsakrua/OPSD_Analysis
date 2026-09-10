#!/bin/bash
# Submit GRPO→OPSD (jsd_clip=0.01) train, write manifest, then start CPU watcher.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/qwen3_4b_grpo/jsd001/submit_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/4b_grpo_jsd001"
mkdir -p "${OUT_DIR}"
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

TRAIN_SCRIPT="scripts/train/qwen3_4b_grpo/jsd001/opsd_student_think_teacher_think_clip001_1e_6_openthoughts_grpo.sh"
MODEL_KEY="qwen3_4b_base"
MODEL_TAG="qwen3_4b_base"
RUN_NAME="st_tt_clip001_1e_6_openthoughts_4b_grpo"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[error] missing ${TRAIN_SCRIPT}" >&2
  exit 1
fi

{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript"
} >"${MANIFEST_STAMPED}"

echo "[submit] root=${ROOT}"
jid=$(sbatch --parsable --chdir="${ROOT}" "${TRAIN_SCRIPT}")
echo -e "${MODEL_KEY}\t${MODEL_TAG}\t${RUN_NAME}\t${jid}\t${TRAIN_SCRIPT}" >>"${MANIFEST_STAMPED}"
echo "[submit] ${MODEL_KEY} -> jid=${jid} script=${TRAIN_SCRIPT}"

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"
echo "[submit] stamped=${MANIFEST_STAMPED}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${ROOT}/scripts/train/qwen3_4b_grpo/jsd001/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
