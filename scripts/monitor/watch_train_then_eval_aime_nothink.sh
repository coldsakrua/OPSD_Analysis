#!/bin/bash
# Watch a training Slurm job; after it finishes, submit AIME24/25/26 nothink evals
# on checkpoint-100.
#
# Usage:
#   bash scripts/monitor/watch_train_then_eval_aime_nothink.sh [TRAIN_JOB_ID]
#   # or:
#   TRAIN_JOB_ID=2944387 nohup bash scripts/monitor/watch_train_then_eval_aime_nothink.sh \
#     > log/monitor/watch_2944387.log 2>&1 &
set -euo pipefail

TRAIN_JOB_ID=${1:-${TRAIN_JOB_ID:-2944387}}
POLL_SEC=${POLL_SEC:-60}
CKPT_WAIT_SEC=${CKPT_WAIT_SEC:-30}
CKPT_READY_TIMEOUT_SEC=${CKPT_READY_TIMEOUT_SEC:-1800}

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUN_NAME=${RUN_NAME:-snt_tt_1e_6}
STEP=${STEP:-100}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/outputs/${RUN_NAME}/${TRAIN_JOB_ID}}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${OUTPUT_DIR}/checkpoint-${STEP}}
EVAL_TAG=${EVAL_TAG:-${RUN_NAME}_ckpt${STEP}}

EVAL_SCRIPTS=(
  "${BASE_DIR}/scripts/eval/4b/aime24_nothink.sh"
  "${BASE_DIR}/scripts/eval/4b/aime25_nothink.sh"
  "${BASE_DIR}/scripts/eval/4b/aime26_nothink.sh"
)

log() {
  echo "[$(date '+%F %T')] $*"
}

job_in_queue() {
  squeue -h -j "${TRAIN_JOB_ID}" 2>/dev/null | grep -q .
}

job_state() {
  # Prefer sacct after the job leaves the queue.
  local st
  st=$(sacct -j "${TRAIN_JOB_ID}" -n -X -o State --parsable2 2>/dev/null | head -n1 | tr -d '[:space:]')
  if [[ -n "${st}" ]]; then
    echo "${st}"
    return
  fi
  st=$(squeue -h -j "${TRAIN_JOB_ID}" -o '%T' 2>/dev/null | head -n1 | tr -d '[:space:]')
  echo "${st:-UNKNOWN}"
}

ckpt_ready() {
  local ckpt="$1"
  [[ -f "${ckpt}/config.json" ]] || return 1
  # HF shard index or single-file weights.
  [[ -f "${ckpt}/model.safetensors.index.json" || -f "${ckpt}/model.safetensors" || -f "${ckpt}/pytorch_model.bin" ]] || return 1
  # Prefer tokenizer present as well.
  [[ -f "${ckpt}/tokenizer.json" || -f "${ckpt}/tokenizer_config.json" ]] || return 1
  return 0
}

wait_for_ckpt() {
  local ckpt="$1"
  local waited=0
  log "waiting for checkpoint: ${ckpt}"
  while ! ckpt_ready "${ckpt}"; do
    if (( waited >= CKPT_READY_TIMEOUT_SEC )); then
      log "ERROR: checkpoint not ready after ${CKPT_READY_TIMEOUT_SEC}s: ${ckpt}"
      ls -la "${OUTPUT_DIR}" 2>/dev/null || true
      return 1
    fi
    sleep "${CKPT_WAIT_SEC}"
    waited=$((waited + CKPT_WAIT_SEC))
    log "  still waiting... (${waited}s)"
  done
  log "checkpoint ready: ${ckpt}"
}

submit_evals() {
  local script ds job_name job_id
  mkdir -p "${BASE_DIR}/log/monitor"
  cd "${BASE_DIR}"

  log "submitting evals: CHECKPOINT_PATH=${CHECKPOINT_PATH} EVAL_TAG=${EVAL_TAG}"
  for script in "${EVAL_SCRIPTS[@]}"; do
    [[ -f "${script}" ]] || { log "ERROR: missing ${script}"; return 1; }
    ds=$(basename "${script}" | sed -E 's/_nothink\.sh$//; s/_think\.sh$//')
    job_name="${RUN_NAME}_${ds}_nt"
    job_id=$(
      sbatch --parsable \
        --job-name="${job_name}" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}" \
        "${script}"
    )
    log "submitted ${ds}: job ${job_id} (${job_name})"
    echo "${job_id}" >> "${BASE_DIR}/log/monitor/eval_jobs_${TRAIN_JOB_ID}.txt"
  done
}

main() {
  log "monitor start: train_job=${TRAIN_JOB_ID} poll=${POLL_SEC}s"
  log "expect ckpt=${CHECKPOINT_PATH}"
  log "eval_tag=${EVAL_TAG}"

  if job_in_queue; then
    log "train job ${TRAIN_JOB_ID} is in queue; watching..."
    while job_in_queue; do
      st=$(squeue -h -j "${TRAIN_JOB_ID}" -o '%T %M %N' 2>/dev/null | head -n1 || true)
      log "  status: ${st:-running}"
      sleep "${POLL_SEC}"
    done
    log "train job left the queue"
  else
    log "train job ${TRAIN_JOB_ID} not in queue; checking final state"
  fi

  # Give filesystem a moment after job exit.
  sleep 5
  st=$(job_state)
  log "final train state: ${st}"
  case "${st}" in
    COMPLETED|COMPLETING|"")
      ;;
    FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY|BOOT_FAIL)
      log "ERROR: train job ended with state=${st}; refuse to submit evals"
      exit 1
      ;;
    *)
      # Some clusters report COMPLETED with trailing + or similar.
      if [[ "${st}" == COMPLETED* ]]; then
        :
      else
        log "WARN: unexpected state=${st}; continue if checkpoint exists"
      fi
      ;;
  esac

  wait_for_ckpt "${CHECKPOINT_PATH}"
  submit_evals
  log "done. eval job ids in log/monitor/eval_jobs_${TRAIN_JOB_ID}.txt"
}

main "$@"
