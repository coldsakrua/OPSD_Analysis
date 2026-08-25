#!/bin/bash
# Submit the three missing Qwen3-1.7B jsd005 think-combo trains and start the monitor.
#
#   snt_tnt → nothink eval
#   snt_tt  → nothink eval
#   st_tnt  → think eval
#
# Usage:
#   bash scripts/monitor/submit_1p7b_jsd005_think_combo_train_eval.sh
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
TRAIN_DIR="${BASE_DIR}/scripts/train/qwen3_1.7b/jsd005"
MONITOR="${BASE_DIR}/scripts/monitor/watch_1p7b_jsd005_think_combo_train_eval.sh"
MODEL_TAG=qwen3_1.7b
EVAL_SUBDIR=1.7b

log() {
  echo "[$(date '+%F %T')] $*"
}

submit_train() {
  local script="$1"
  local jid
  jid=$(sbatch --parsable "${script}" | tail -1)
  if [[ -z "${jid}" ]]; then
    log "ERROR: sbatch failed for ${script}" >&2
    exit 1
  fi
  log "submitted train ${script##*/}: job ${jid}" >&2
  echo "${jid}"
}

cd "${BASE_DIR}"
mkdir -p log/monitor

JID_SNT_TNT=$(submit_train "${TRAIN_DIR}/opsd_student_nothink_teacher_nothink_clip005_1e_6_openthoughts.sh")
JID_SNT_TT=$(submit_train "${TRAIN_DIR}/opsd_student_nothink_teacher_think_clip005_1e_6_openthoughts.sh")
JID_ST_TNT=$(submit_train "${TRAIN_DIR}/opsd_student_think_teacher_nothink_clip005_1e_6_openthoughts.sh")

SPECS=(
  "${JID_SNT_TNT}|${MODEL_TAG}|snt_tnt_clip005_1e_6_openthoughts_1p7b|snt_tnt_clip005_1e6_1p7b_ckpt100|snt_tnt_clip005_1e6_1p7b|${EVAL_SUBDIR}|nothink|nt"
  "${JID_SNT_TT}|${MODEL_TAG}|snt_tt_clip005_1e_6_openthoughts_1p7b|snt_tt_clip005_1e6_1p7b_ckpt100|snt_tt_clip005_1e6_1p7b|${EVAL_SUBDIR}|nothink|nt"
  "${JID_ST_TNT}|${MODEL_TAG}|st_tnt_clip005_1e_6_openthoughts_1p7b|st_tnt_clip005_1e6_1p7b_ckpt100|st_tnt_clip005_1e6_1p7b|${EVAL_SUBDIR}|think|th"
)

MON_LOG="${BASE_DIR}/log/monitor/watch_1p7b_jsd005_think_combo_$(date +%Y%m%d_%H%M%S).log"
SUBMIT_LOG="${BASE_DIR}/log/monitor/submit_1p7b_jsd005_think_combo_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "train_jobs:"
  echo "  snt_tnt=${JID_SNT_TNT}"
  echo "  snt_tt=${JID_SNT_TT}"
  echo "  st_tnt=${JID_ST_TNT}"
  echo "monitor_log=${MON_LOG}"
  echo "specs:"
  printf '  %s\n' "${SPECS[@]}"
} | tee "${SUBMIT_LOG}"

SPECS_OVERRIDE="${SPECS[*]}" nohup bash "${MONITOR}" > "${MON_LOG}" 2>&1 &
MON_PID=$!

log "monitor pid=${MON_PID} log=${MON_LOG}"
log "submit record -> ${SUBMIT_LOG}"
