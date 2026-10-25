#!/bin/bash
#SBATCH --job-name=watch_omr_sol_8b_50
#SBATCH --output=log/train/omr_solution_postthink_50step/watch_8b_slurm.%j.out
#SBATCH --partition=C64M512G,C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
set -euo pipefail

# Watch Qwen3-8B OMR-solution (postthink) st_tt 50-step OPSD. On complete
# checkpoint-50, submit 4 think evals (aime24/25/26 + hmmt25).
#
# Required: TRAIN_JID
# Optional: RUN_NAME, MODEL_TAG, CKPT_STEP, INTERVAL, POST_DONE_WAIT

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
cd "${BASE_DIR}"

: "${TRAIN_JID:?set via sbatch --export}"
RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_omr_solution_postthink_50step_8b}
MODEL_TAG=${MODEL_TAG:-qwen3_8b}
CKPT_STEP=${CKPT_STEP:-50}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}

STATE_DIR="${BASE_DIR}/log/train/omr_solution_postthink_50step/watch_state"
mkdir -p "${STATE_DIR}" "log/train/omr_solution_postthink_50step"

log() { echo "[$(date -Is)] $*"; }

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
}

find_ckpt() {
  local root="${BASE_DIR}/outputs/${MODEL_TAG}/${RUN_NAME}/${TRAIN_JID}"
  if ckpt_complete "${root}/checkpoint-${CKPT_STEP}"; then
    echo "${root}/checkpoint-${CKPT_STEP}"
    return 0
  fi
  if ckpt_complete "${root}/final"; then
    echo "${root}/final"
    return 0
  fi
  return 1
}

stamp="${STATE_DIR}/8b_${TRAIN_JID}.done"
fail_stamp="${STATE_DIR}/8b_${TRAIN_JID}.failed"

log "host=$(hostname) job=${SLURM_JOB_ID:-manual} interval=${INTERVAL}s ckpt_step=${CKPT_STEP}"
log "watching train_jid=${TRAIN_JID} run=${RUN_NAME} model_tag=${MODEL_TAG}"

while true; do
  if [[ -f "${stamp}" ]]; then
    log "already submitted; exiting"
    break
  fi

  if ckpt=$(find_ckpt); then
    log "READY jid=${TRAIN_JID} -> ${ckpt} (train_state=$(job_state "${TRAIN_JID}"))"
    sleep "${POST_DONE_WAIT}"
    if ! ckpt=$(find_ckpt); then
      log "WAIT ckpt disappeared after settle"
      sleep "${INTERVAL}"
      continue
    fi
    eval_tag="${RUN_NAME}_ckpt${CKPT_STEP}"
    if [[ "$(basename "${ckpt}")" != "checkpoint-${CKPT_STEP}" ]]; then
      eval_tag="${RUN_NAME}_final"
    fi
    if CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${eval_tag}" SEED=42 BASE_DIR="${BASE_DIR}" \
        bash "${BASE_DIR}/scripts/eval/8b/submit_four.sh"; then
      date -Is >"${stamp}"
      echo "${ckpt}" >>"${stamp}"
      rm -f "${fail_stamp}"
      log "SUBMITTED 4 think evals tag=${eval_tag}"
      break
    fi
    log "ERROR submitting evals; will retry"
    sleep "${INTERVAL}"
    continue
  fi

  if job_in_queue "${TRAIN_JID}"; then
    log "WAIT jid=${TRAIN_JID} state=$(job_state "${TRAIN_JID}")"
    sleep "${INTERVAL}"
    continue
  fi

  st=$(job_state "${TRAIN_JID}")
  case "${st}" in
    FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
      log "FAIL jid=${TRAIN_JID} state=${st}, no usable checkpoint-${CKPT_STEP}/final"
      date -Is >"${fail_stamp}"
      echo "${st}" >>"${fail_stamp}"
      exit 1
      ;;
    *)
      log "WAIT jid=${TRAIN_JID} left queue (state=${st}) but no usable ckpt yet"
      sleep "${INTERVAL}"
      ;;
  esac
done

log "watch finished"
exit 0
