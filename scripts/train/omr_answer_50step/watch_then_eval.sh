#!/bin/bash
#SBATCH --job-name=watch_omr_answer_50step
#SBATCH --output=log/train/omr_answer_50step/watch_slurm.%j.out
#SBATCH --partition=C64M256G,C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
set -euo pipefail

# CPU-node watcher for the OMR answer-privilege 50-step OPSD runs (1.7B + 4B).
# When a train job produces a complete checkpoint-50 (or final), submit the
# 4 think evals (aime24/25/26 + hmmt25) for that model.
#
# Required env (via sbatch --export):
#   TRAIN_JID_17, TRAIN_JID_4
# Optional:
#   RUN_NAME_17 / RUN_NAME_4, CKPT_STEP (default 50), INTERVAL, POST_DONE_WAIT

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
cd "${BASE_DIR}"

: "${TRAIN_JID_17:?set via sbatch --export}"
: "${TRAIN_JID_4:?set via sbatch --export}"
RUN_NAME_17=${RUN_NAME_17:-st_tt_clip005_1e_6_omr_answer_50step_1p7b}
RUN_NAME_4=${RUN_NAME_4:-st_tt_clip005_1e_6_omr_answer_50step_4b}
CKPT_STEP=${CKPT_STEP:-50}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}

STATE_DIR="${BASE_DIR}/log/train/omr_answer_50step/watch_state"
mkdir -p "${STATE_DIR}"

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
  local model_tag=$1 run_name=$2 train_jid=$3
  local root="${BASE_DIR}/outputs/${model_tag}/${run_name}/${train_jid}"
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

# Returns 0 when this model is fully handled (evals submitted), 1 otherwise.
handle_one() {
  local model_key=$1 model_tag=$2 run_name=$3 train_jid=$4
  local eval_tag="${run_name}_ckpt${CKPT_STEP}"
  local stamp="${STATE_DIR}/${model_key}_${train_jid}.done"
  local fail_stamp="${STATE_DIR}/${model_key}_${train_jid}.failed"

  if [[ -f "${stamp}" ]]; then
    return 0
  fi

  local ckpt=""
  if ckpt=$(find_ckpt "${model_tag}" "${run_name}" "${train_jid}"); then
    log "READY ${model_key} jid=${train_jid} -> ${ckpt} (train_state=$(job_state "${train_jid}"))"
    sleep "${POST_DONE_WAIT}"
    if ! ckpt=$(find_ckpt "${model_tag}" "${run_name}" "${train_jid}"); then
      log "WAIT ${model_key} jid=${train_jid} (ckpt disappeared after wait)"
      return 1
    fi
    if [[ "$(basename "${ckpt}")" != "checkpoint-${CKPT_STEP}" ]]; then
      eval_tag="${run_name}_final"
    fi
    if CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${eval_tag}" SEED=42 BASE_DIR="${BASE_DIR}" \
        bash "${BASE_DIR}/scripts/eval/${model_key}/submit_four.sh"; then
      date -Is >"${stamp}"
      echo "${ckpt}" >>"${stamp}"
      rm -f "${fail_stamp}"
      log "SUBMITTED 4 evals for ${model_key} tag=${eval_tag}"
      return 0
    fi
    log "ERROR submitting evals for ${model_key} (will retry)"
    return 1
  fi

  if job_in_queue "${train_jid}"; then
    log "WAIT ${model_key} jid=${train_jid} state=$(job_state "${train_jid}")"
    return 1
  fi

  local st
  st=$(job_state "${train_jid}")
  case "${st}" in
    FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
      log "FAIL ${model_key} jid=${train_jid} state=${st}, no usable checkpoint-${CKPT_STEP}/final"
      date -Is >"${fail_stamp}"
      echo "${st}" >>"${fail_stamp}"
      return 0  # handled (failed); do not block the other model
      ;;
    *)
      log "WAIT ${model_key} jid=${train_jid} left queue (state=${st}) but no usable ckpt yet"
      return 1
      ;;
  esac
}

log "host=$(hostname) job=${SLURM_JOB_ID:-manual} interval=${INTERVAL}s ckpt_step=${CKPT_STEP}"
log "watching 1.7b jid=${TRAIN_JID_17} run=${RUN_NAME_17} | 4b jid=${TRAIN_JID_4} run=${RUN_NAME_4}"

while true; do
  done_count=0
  handle_one "1.7b" "qwen3_1.7b" "${RUN_NAME_17}" "${TRAIN_JID_17}" && done_count=$((done_count + 1))
  handle_one "4b" "qwen3_4b" "${RUN_NAME_4}" "${TRAIN_JID_4}" && done_count=$((done_count + 1))
  log "progress done=${done_count}/2"
  if [[ "${done_count}" -ge 2 ]]; then
    log "ALL trains handled; exiting"
    break
  fi
  sleep "${INTERVAL}"
done

log "watch finished"
exit 0
