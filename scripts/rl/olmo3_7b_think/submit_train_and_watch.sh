#!/bin/bash
# Submit Olmo-3-7B-Think GRPO train, write watch manifest (target gs100),
# then start a CPU-node watcher: merge FSDP → 4 think evals.
#
# Usage (from OPSD_Analysis):
#   bash scripts/rl/olmo3_7b_think/submit_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/rl/olmo3_7b_think"
mkdir -p "${OUT_DIR}"
MANIFEST="${OUT_DIR}/watch_grpo_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/watch_grpo_${STAMP}.tsv"

TRAIN_SCRIPT="scripts/rl/olmo3_7b_think/grpo_think_olmo7bt_omr_int_6gpu_bs64_24k_6400.sh"
WATCH_SCRIPT="scripts/rl/olmo3_7b_think/sbatch_watch.sh"
MODEL_KEY="olmo3_7b_think"
MODEL_TAG="olmo3_7b_think"
RUN_NAME="grpo_think_olmo7bt_omr_int_6gpu_bs64_24k_6400"
TARGET_STEP=100
SEED=42
EVAL_TAG="grpo_omr_int_olmo7bt_gs100"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[error] missing ${TRAIN_SCRIPT}" >&2
  exit 1
fi

chmod +x "${TRAIN_SCRIPT}" "${WATCH_SCRIPT}" \
  "${ROOT}/scripts/rl/olmo3_7b_think/watch_then_eval.sh"

{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\ttarget_step\tseed\teval_tag"
} >"${MANIFEST_STAMPED}"

echo "[submit] root=${ROOT}"
jid=$(sbatch --parsable --chdir="${ROOT}" "${TRAIN_SCRIPT}")
jid="${jid%%;*}"
echo -e "${MODEL_KEY}\t${MODEL_TAG}\t${RUN_NAME}\t${jid}\t${TARGET_STEP}\t${SEED}\t${EVAL_TAG}" >>"${MANIFEST_STAMPED}"
echo "[submit] train ${MODEL_KEY} -> jid=${jid} script=${TRAIN_SCRIPT}"

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"
echo "[submit] stamped=${MANIFEST_STAMPED}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${WATCH_SCRIPT}")
watch_jid="${watch_jid%%;*}"
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
