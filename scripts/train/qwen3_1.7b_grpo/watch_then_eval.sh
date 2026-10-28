#!/bin/bash
#SBATCH --job-name=watch_1p7b_grpo_opsd
#SBATCH --output=log/train/1.7b_grpo/watch_slurm.%j.out
#SBATCH --partition=C64M256G,C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
set -euo pipefail

# Watch 1.7B-Base SFT→GRPO→OPSD train; on checkpoint-100 submit 4 think evals
# (aime24 / aime25 / aime26 / hmmt25) via scripts/eval/qwen3_1.7b_base/submit_four.sh.
#
# Required via sbatch --export: TRAIN_JID
# Optional: RUN_NAME, MODEL_TAG, CKPT_STEP, INTERVAL, POST_DONE_WAIT, EVAL_SEED

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
cd "${BASE_DIR}"

: "${TRAIN_JID:?set via sbatch --export}"

RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_openthoughts_1p7b_grpo}
MODEL_TAG=${MODEL_TAG:-qwen3_1.7b_base}
CKPT_STEP=${CKPT_STEP:-100}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}
EVAL_SEED=${EVAL_SEED:-42}
EVAL_TAG=${EVAL_TAG:-${RUN_NAME}_ckpt${CKPT_STEP}}

STATE_DIR="${BASE_DIR}/log/train/1.7b_grpo/watch_state"
SUBMIT="${BASE_DIR}/scripts/eval/qwen3_1.7b_base/submit_four.sh"
mkdir -p "${STATE_DIR}" "log/train/1.7b_grpo"
chmod +x "${SUBMIT}" 2>/dev/null || true

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

stamp="${STATE_DIR}/${EVAL_TAG}_${TRAIN_JID}.done"
fail_stamp="${STATE_DIR}/${EVAL_TAG}_${TRAIN_JID}.failed"

log "watching train_jid=${TRAIN_JID} run=${RUN_NAME} ckpt=${CKPT_STEP}"
log "outputs → ${BASE_DIR}/outputs/${MODEL_TAG}/${RUN_NAME}/${TRAIN_JID}"

while true; do
  if [[ -f "${stamp}" ]]; then
    log "already submitted; exiting"
    break
  fi

  if ckpt=$(find_ckpt); then
    log "READY jid=${TRAIN_JID} -> ${ckpt} (train_state=$(job_state "${TRAIN_JID}"))"
    sleep "${POST_DONE_WAIT}"
    if ! ckpt=$(find_ckpt); then
      log "WAIT ckpt disappeared after wait; retry"
      sleep "${INTERVAL}"
      continue
    fi
    submit_tag="${EVAL_TAG}"
    ckpt_base="$(basename "${ckpt}")"
    if [[ "${ckpt_base}" != "checkpoint-${CKPT_STEP}" ]]; then
      submit_tag="${RUN_NAME}_${ckpt_base}"
    fi
    if CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${submit_tag}" SEED="${EVAL_SEED}" BASE_DIR="${BASE_DIR}" \
        bash "${SUBMIT}"; then
      date -Is >"${stamp}"
      echo "${ckpt}" >>"${stamp}"
      rm -f "${fail_stamp}"
      log "SUBMITTED 4 evals tag=${submit_tag} seed=${EVAL_SEED}"
      break
    fi
    log "ERROR submit_four failed; will retry"
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
      log "FAIL jid=${TRAIN_JID} state=${st} and no usable ckpt"
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
