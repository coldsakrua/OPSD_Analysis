#!/bin/bash
# Submit Qwen3-4B jsd005 snt_tnt (solution) train and start monitor → nothink eval.
#
# Usage:
#   bash scripts/monitor/submit_4b_snt_tnt_train_eval.sh
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
TRAIN_SCRIPT="${BASE_DIR}/scripts/train/qwen3_4b/jsd005/opsd_student_nothink_teacher_nothink_clip005_1e_6_openthoughts.sh"
MONITOR="${BASE_DIR}/scripts/monitor/watch_le80_train_eval_ckpt100.sh"
MODEL_TAG=qwen3_4b
RUN_NAME=snt_tnt_clip005_1e_6_openthoughts_4b
EVAL_TAG=snt_tnt_clip005_1e6_4b_ckpt100
PREFIX=snt_tnt_clip005_4b
EVAL_SUBDIR=4b

log() {
  echo "[$(date '+%F %T')] $*" >&2
}

cd "${BASE_DIR}"
mkdir -p log/monitor

JID=$(sbatch --parsable "${TRAIN_SCRIPT}" | tail -1)
if [[ -z "${JID}" ]]; then
  log "ERROR: sbatch failed for ${TRAIN_SCRIPT}"
  exit 1
fi
log "submitted train: job ${JID}"

SPEC="${JID}|${MODEL_TAG}|${RUN_NAME}|${EVAL_TAG}|${PREFIX}|${EVAL_SUBDIR}|nothink|nt"
MON_LOG="${BASE_DIR}/log/monitor/watch_4b_snt_tnt_$(date +%Y%m%d_%H%M%S).log"
SUBMIT_LOG="${BASE_DIR}/log/monitor/submit_4b_snt_tnt_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "train_job=${JID}"
  echo "run_name=${RUN_NAME}"
  echo "eval_mode=nothink"
  echo "monitor_log=${MON_LOG}"
  echo "spec=${SPEC}"
} | tee "${SUBMIT_LOG}"

SPECS_OVERRIDE="${SPEC}" nohup bash "${MONITOR}" > "${MON_LOG}" 2>&1 &
MON_PID=$!

log "monitor pid=${MON_PID} log=${MON_LOG}"
log "submit record -> ${SUBMIT_LOG}"
