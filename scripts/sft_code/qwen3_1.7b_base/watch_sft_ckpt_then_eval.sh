#!/bin/bash
# Watch one SFT run until checkpoint-${CKPT_STEP} is complete, then submit
# LCB / HumanEval+ / MBPP+ generate jobs (judges auto-submitted).
#
# Resume jobs write into the original OUTPUT_DIR (e.g. .../3925961) while the
# live Slurm jid may differ (e.g. 4071750). Pass both:
#   TRAIN_JID=4071750 OUTPUT_DIR=.../3925961 CKPT_STEP=15000
#
# Usage:
#   bash scripts/sft_code/qwen3_1.7b_base/watch_sft_ckpt_then_eval.sh
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
cd "${BASE_DIR}"

TRAIN_JID=${TRAIN_JID:-4071750}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/outputs/qwen3_1.7b_base/sft_think_4gpu_12k_ocr/3925961}
CKPT_STEP=${CKPT_STEP:-15000}
MODEL_KEY=${MODEL_KEY:-qwen3_1.7b_base}
EVAL_TAG=${EVAL_TAG:-sft_think_4gpu_12k_ocr_ckpt${CKPT_STEP}}
INTERVAL=${INTERVAL:-60}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}
SEED=${SEED:-42}
AUTO_JUDGE=${AUTO_JUDGE:-1}

SUBMIT="${BASE_DIR}/scripts/train_code/ocr/submit_three_code_evals.sh"
STATE_DIR="${BASE_DIR}/log/train_code/ocr/watch_state"
LOG="${BASE_DIR}/log/train_code/sft/1.7b_base/watch_ckpt${CKPT_STEP}.${SLURM_JOB_ID:-manual}.log"
DONE_STAMP="${STATE_DIR}/${EVAL_TAG}_${TRAIN_JID}.done"
FAIL_STAMP="${STATE_DIR}/${EVAL_TAG}_${TRAIN_JID}.failed"
mkdir -p "${STATE_DIR}" "$(dirname "${LOG}")"
chmod +x "${SUBMIT}" 2>/dev/null || true

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

job_in_queue() {
  local jid=$1
  [[ -n "$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)" ]]
}

job_state() {
  local jid=$1 st
  st=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
  if [[ -n "${st}" ]]; then
    echo "${st}"
    return 0
  fi
  sacct -j "${jid}" -n -X -o State -P 2>/dev/null | head -1 | tr -d ' ' || echo UNKNOWN
}

ckpt_has_weights() {
  local p=$1
  [[ -f "${p}/config.json" ]] || return 1
  [[ -f "${p}/model.safetensors" ]] \
    || [[ -f "${p}/model.safetensors.index.json" ]] \
    || [[ -f "${p}/model-00001-of-00001.safetensors" ]] \
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

ckpt_complete() {
  local p=$1
  ckpt_has_weights "${p}" || return 1
  [[ -f "${p}/trainer_state.json" ]] || return 1
  python3 - "${p}" "${CKPT_STEP}" <<'PY' 2>/dev/null || return 1
import json, sys
p, step = sys.argv[1], int(sys.argv[2])
d = json.load(open(f"{p}/trainer_state.json"))
gs = int(d.get("global_step") or 0)
sys.exit(0 if gs >= step else 1)
PY
}

find_ckpt() {
  local p="${OUTPUT_DIR}/checkpoint-${CKPT_STEP}"
  if ckpt_complete "${p}"; then
    echo "${p}"
    return 0
  fi
  return 1
}

if [[ -f "${DONE_STAMP}" ]]; then
  log "SKIP already submitted: ${DONE_STAMP}"
  exit 0
fi

log "host=$(hostname) watch_job=${SLURM_JOB_ID:-none}"
log "train_jid=${TRAIN_JID} output_dir=${OUTPUT_DIR}"
log "ckpt_step=${CKPT_STEP} eval_tag=${EVAL_TAG} interval=${INTERVAL}s"

while true; do
  if [[ -f "${DONE_STAMP}" ]]; then
    log "SKIP done stamp appeared"
    exit 0
  fi

  if ckpt=$(find_ckpt); then
    st=$(job_state "${TRAIN_JID}")
    log "READY ckpt=${ckpt} train_state=${st}; waiting ${POST_DONE_WAIT}s for flush"
    sleep "${POST_DONE_WAIT}"
    if ! ckpt=$(find_ckpt); then
      log "WAIT ckpt disappeared after wait; retry"
      sleep "${INTERVAL}"
      continue
    fi
    if MODEL_KEY="${MODEL_KEY}" CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${EVAL_TAG}" \
        SEED="${SEED}" AUTO_JUDGE="${AUTO_JUDGE}" BASE_DIR="${BASE_DIR}" \
        bash "${SUBMIT}" >>"${LOG}" 2>&1; then
      date -Is >"${DONE_STAMP}"
      echo "${ckpt}" >>"${DONE_STAMP}"
      rm -f "${FAIL_STAMP}"
      log "SUBMITTED evals tag=${EVAL_TAG} ckpt=${ckpt}"
      exit 0
    fi
    log "ERROR submit failed; will retry"
    sleep "${INTERVAL}"
    continue
  fi

  if job_in_queue "${TRAIN_JID}"; then
    st=$(job_state "${TRAIN_JID}")
    log "WAIT train_jid=${TRAIN_JID} state=${st} (no ckpt-${CKPT_STEP} yet)"
    sleep "${INTERVAL}"
    continue
  fi

  st=$(job_state "${TRAIN_JID}")
  case "${st}" in
    COMPLETED|COMPLETING)
      log "WAIT train finished (${st}) but ckpt-${CKPT_STEP} not ready yet"
      ;;
    FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
      log "FAIL train_jid=${TRAIN_JID} state=${st} and no usable ckpt-${CKPT_STEP}"
      date -Is >"${FAIL_STAMP}"
      echo "${st}" >>"${FAIL_STAMP}"
      exit 1
      ;;
    *)
      log "WAIT train left queue state=${st}; still looking for ckpt-${CKPT_STEP}"
      ;;
  esac
  sleep "${INTERVAL}"
done
