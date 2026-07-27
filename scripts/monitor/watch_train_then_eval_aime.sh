#!/bin/bash
# Watch train job; submit AIME24/25/26 think+nothink evals once checkpoint-100
# is ready, or when the job ends (including FAILED) if the ckpt already exists.
#
# Usage:
#   nohup bash scripts/monitor/watch_train_then_eval_aime.sh 2948221 \
#     > log/monitor/watch_2948221.log 2>&1 &
#
# Optional env overrides:
#   RUN_NAME  OUTPUT_DIR  CHECKPOINT_PATH  EVAL_TAG  POLL_SEC  STEP
set -euo pipefail

TRAIN_JOB_ID=${1:-${TRAIN_JOB_ID:?Set TRAIN_JOB_ID or pass job id as \$1}}
POLL_SEC=${POLL_SEC:-30}
CKPT_READY_TIMEOUT_SEC=${CKPT_READY_TIMEOUT_SEC:-3600}
STEP=${STEP:-100}

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
# Job name is snt_tt_1e6_ota; train RUN_NAME is snt_tt_1e_6_openthoughts_answer.
RUN_NAME=${RUN_NAME:-snt_tt_1e_6_openthoughts_answer}
EVAL_TAG=${EVAL_TAG:-snt_tt_1e6_ota_ckpt${STEP}}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-snt_tt_1e6_ota}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/outputs/${RUN_NAME}/${TRAIN_JOB_ID}}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${OUTPUT_DIR}/checkpoint-${STEP}}

EVAL_SCRIPTS=(
  "${BASE_DIR}/scripts/eval/4b/aime24_nothink.sh"
  "${BASE_DIR}/scripts/eval/4b/aime24_think.sh"
  "${BASE_DIR}/scripts/eval/4b/aime25_nothink.sh"
  "${BASE_DIR}/scripts/eval/4b/aime25_think.sh"
  "${BASE_DIR}/scripts/eval/4b/aime26_nothink.sh"
  "${BASE_DIR}/scripts/eval/4b/aime26_think.sh"
)

LOCK_FILE="${BASE_DIR}/log/monitor/eval_submitted_${TRAIN_JOB_ID}.lock"
EVAL_JOBS_FILE="${BASE_DIR}/log/monitor/eval_jobs_${TRAIN_JOB_ID}.txt"

log() {
  echo "[$(date '+%F %T')] $*"
}

job_in_queue() {
  squeue -h -j "${TRAIN_JOB_ID}" 2>/dev/null | grep -q .
}

job_state() {
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
  [[ -d "${ckpt}" ]] || return 1
  [[ -f "${ckpt}/config.json" ]] || return 1
  [[ -f "${ckpt}/model.safetensors.index.json" || -f "${ckpt}/model.safetensors" || -f "${ckpt}/pytorch_model.bin" ]] || return 1
  [[ -f "${ckpt}/tokenizer.json" || -f "${ckpt}/tokenizer_config.json" ]] || return 1
  return 0
}

mode_suffix() {
  local script="$1"
  case "$(basename "${script}")" in
    *_nothink.sh) echo nt ;;
    *_think.sh) echo th ;;
    *) echo eval ;;
  esac
}

dataset_tag() {
  local script="$1"
  basename "${script}" | sed -E 's/_(nothink|think)\.sh$//'
}

submit_evals() {
  local script ds mode job_name job_id
  mkdir -p "${BASE_DIR}/log/monitor"
  if [[ -f "${LOCK_FILE}" ]]; then
    log "evals already submitted (lock=${LOCK_FILE}); skip"
    return 0
  fi
  : > "${LOCK_FILE}"
  : > "${EVAL_JOBS_FILE}"

  cd "${BASE_DIR}"
  log "submitting 6 evals: CHECKPOINT_PATH=${CHECKPOINT_PATH} EVAL_TAG=${EVAL_TAG}"
  for script in "${EVAL_SCRIPTS[@]}"; do
    [[ -f "${script}" ]] || { log "ERROR: missing ${script}"; return 1; }
    ds=$(dataset_tag "${script}")
    mode=$(mode_suffix "${script}")
    # e.g. snt_tt_1e6_ota_a24_nt
    job_name="${JOB_NAME_PREFIX}_${ds/aime/a}_${mode}"
    job_id=$(
      sbatch --parsable \
        --job-name="${job_name}" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}" \
        "${script}" 2>/dev/null | tail -1
    )
    log "submitted ${ds}_${mode}: job ${job_id} (${job_name})"
    echo "${job_id} ${job_name}" >> "${EVAL_JOBS_FILE}"
  done
  log "done. job ids -> ${EVAL_JOBS_FILE}"
}

main() {
  mkdir -p "${BASE_DIR}/log/monitor"
  log "monitor start: train_job=${TRAIN_JOB_ID} poll=${POLL_SEC}s"
  log "output_dir=${OUTPUT_DIR}"
  log "expect ckpt=${CHECKPOINT_PATH}"
  log "eval_tag=${EVAL_TAG}"

  local waited=0
  while true; do
    if ckpt_ready "${CHECKPOINT_PATH}"; then
      log "checkpoint-${STEP} ready"
      submit_evals
      exit 0
    fi

    if ! job_in_queue; then
      local st
      st=$(job_state)
      log "train job left queue; state=${st}"
      # Give FS a moment if save was in flight at crash.
      sleep 10
      if ckpt_ready "${CHECKPOINT_PATH}"; then
        log "checkpoint found after job end; submitting evals"
        submit_evals
        exit 0
      fi
      log "ERROR: no usable checkpoint-${STEP} after job end (${st})"
      ls -la "${OUTPUT_DIR}" 2>/dev/null || true
      exit 1
    fi

    local st_line
    st_line=$(squeue -h -j "${TRAIN_JOB_ID}" -o '%T %M %N' 2>/dev/null | head -n1 || true)
    log "waiting... job=[${st_line:-?}] ckpt_missing waited=${waited}s"
    sleep "${POLL_SEC}"
    waited=$((waited + POLL_SEC))
    if (( waited >= CKPT_READY_TIMEOUT_SEC )) && ! job_in_queue; then
      log "ERROR: timeout waiting for ckpt"
      exit 1
    fi
  done
}

main "$@"
