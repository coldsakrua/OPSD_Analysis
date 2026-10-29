#!/bin/bash
#SBATCH --job-name=watch_ot_int_as_mcq_50step
#SBATCH --output=log/train/ot_integer_as_mcq_50step/watch_slurm.%j.out
#SBATCH --partition=C64M256G,C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
set -euo pipefail

# Watch 4 OT-integer-as-MCQ 50-step OPSD trains.
# - 1.7b letter / letter+option: eval as soon as each ckpt-50 is ready.
# - olmo letter / letter+option: wait until BOTH trains are settled, then eval both
#   (avoids burning GPUs on one olmo eval while the sibling is still training/queued).
#
# Required via sbatch --export:
#   TRAIN_JID_17L TRAIN_JID_17O TRAIN_JID_OLL TRAIN_JID_OLO
# Optional: RUN_NAME_*, CKPT_STEP, INTERVAL, POST_DONE_WAIT, EVAL_SEED

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
cd "${BASE_DIR}"

: "${TRAIN_JID_17L:?}"
: "${TRAIN_JID_17O:?}"
: "${TRAIN_JID_OLL:?}"
: "${TRAIN_JID_OLO:?}"

RUN_NAME_17L=${RUN_NAME_17L:-st_tt_clip005_1e_6_ot_integer_as_mcq_letter_50step_1p7b}
RUN_NAME_17O=${RUN_NAME_17O:-st_tt_clip005_1e_6_ot_integer_as_mcq_letter_option_50step_1p7b}
RUN_NAME_OLL=${RUN_NAME_OLL:-st_tt_clip005_1e_6_ot_integer_as_mcq_letter_50step_olmo7bt}
RUN_NAME_OLO=${RUN_NAME_OLO:-st_tt_clip005_1e_6_ot_integer_as_mcq_letter_option_50step_olmo7bt}
CKPT_STEP=${CKPT_STEP:-50}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}
EVAL_SEED=${EVAL_SEED:-42}

STATE_DIR="${BASE_DIR}/log/train/ot_integer_as_mcq_50step/watch_state"
mkdir -p "${STATE_DIR}" "log/train/ot_integer_as_mcq_50step"

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

submit_evals() {
  local eval_family=$1 ckpt=$2 eval_tag=$3
  case "${eval_family}" in
    1.7b)
      CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${eval_tag}" SEED="${EVAL_SEED}" BASE_DIR="${BASE_DIR}" \
        bash "${BASE_DIR}/scripts/eval/1.7b/submit_four.sh"
      ;;
    olmo_7b_think)
      CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${eval_tag}" SEED="${EVAL_SEED}" BASE_DIR="${BASE_DIR}" \
        bash "${BASE_DIR}/scripts/eval/olmo_7b_think/seed/submit_four_seed.sh"
      ;;
    *)
      echo "[error] unknown eval_family=${eval_family}" >&2
      return 1
      ;;
  esac
}

# Returns 0 when this run is fully handled (evals submitted or terminal fail).
handle_one() {
  local key=$1 model_tag=$2 run_name=$3 train_jid=$4 eval_family=$5
  local eval_tag="${run_name}_ckpt${CKPT_STEP}"
  local stamp="${STATE_DIR}/${key}_${train_jid}.done"
  local fail_stamp="${STATE_DIR}/${key}_${train_jid}.failed"

  if [[ -f "${stamp}" ]]; then
    return 0
  fi

  local ckpt=""
  if ckpt=$(find_ckpt "${model_tag}" "${run_name}" "${train_jid}"); then
    log "READY ${key} jid=${train_jid} -> ${ckpt} (train_state=$(job_state "${train_jid}"))"
    sleep "${POST_DONE_WAIT}"
    if ! ckpt=$(find_ckpt "${model_tag}" "${run_name}" "${train_jid}"); then
      log "WAIT ${key} jid=${train_jid} (ckpt disappeared after wait)"
      return 1
    fi
    if [[ "$(basename "${ckpt}")" != "checkpoint-${CKPT_STEP}" ]]; then
      eval_tag="${run_name}_final"
    fi
    if submit_evals "${eval_family}" "${ckpt}" "${eval_tag}"; then
      date -Is >"${stamp}"
      echo "${ckpt}" >>"${stamp}"
      rm -f "${fail_stamp}"
      log "SUBMITTED 4 evals for ${key} tag=${eval_tag} seed=${EVAL_SEED}"
      return 0
    fi
    log "ERROR submitting evals for ${key} (will retry)"
    return 1
  fi

  if job_in_queue "${train_jid}"; then
    log "WAIT ${key} jid=${train_jid} state=$(job_state "${train_jid}")"
    return 1
  fi

  local st
  st=$(job_state "${train_jid}")
  case "${st}" in
    FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
      log "FAIL ${key} jid=${train_jid} state=${st}, no usable checkpoint-${CKPT_STEP}/final"
      date -Is >"${fail_stamp}"
      echo "${st}" >>"${fail_stamp}"
      return 0
      ;;
    *)
      log "WAIT ${key} jid=${train_jid} left queue (state=${st}) but no usable ckpt yet"
      return 1
      ;;
  esac
}

# Probe one train without submitting evals.
# Prints: done | failed | ready:<ckpt> | waiting | missing
# Exit 0 for done|failed|ready; 1 for waiting|missing.
probe_one() {
  local key=$1 model_tag=$2 run_name=$3 train_jid=$4
  local stamp="${STATE_DIR}/${key}_${train_jid}.done"
  local fail_stamp="${STATE_DIR}/${key}_${train_jid}.failed"
  local ckpt=""

  if [[ -f "${stamp}" ]]; then
    echo "done"
    return 0
  fi
  if [[ -f "${fail_stamp}" ]]; then
    echo "failed"
    return 0
  fi
  if ckpt=$(find_ckpt "${model_tag}" "${run_name}" "${train_jid}"); then
    echo "ready:${ckpt}"
    return 0
  fi
  if job_in_queue "${train_jid}"; then
    echo "waiting"
    return 1
  fi
  local st
  st=$(job_state "${train_jid}")
  case "${st}" in
    FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
      echo "failed"
      date -Is >"${fail_stamp}"
      echo "${st}" >>"${fail_stamp}"
      return 0
      ;;
    *)
      echo "missing"
      return 1
      ;;
  esac
}

# Submit eval for a ready ckpt (idempotent via .done stamp).
submit_ready() {
  local key=$1 model_tag=$2 run_name=$3 train_jid=$4 eval_family=$5 ckpt=$6
  local eval_tag="${run_name}_ckpt${CKPT_STEP}"
  local stamp="${STATE_DIR}/${key}_${train_jid}.done"
  if [[ -f "${stamp}" ]]; then
    return 0
  fi
  if [[ "$(basename "${ckpt}")" != "checkpoint-${CKPT_STEP}" ]]; then
    eval_tag="${run_name}_final"
  fi
  log "READY ${key} jid=${train_jid} -> ${ckpt} (paired olmo gate open)"
  sleep "${POST_DONE_WAIT}"
  if ! ckpt=$(find_ckpt "${model_tag}" "${run_name}" "${train_jid}"); then
    log "WAIT ${key} jid=${train_jid} (ckpt disappeared after wait)"
    return 1
  fi
  if [[ "$(basename "${ckpt}")" != "checkpoint-${CKPT_STEP}" ]]; then
    eval_tag="${run_name}_final"
  fi
  if submit_evals "${eval_family}" "${ckpt}" "${eval_tag}"; then
    date -Is >"${stamp}"
    echo "${ckpt}" >>"${stamp}"
    rm -f "${STATE_DIR}/${key}_${train_jid}.failed"
    log "SUBMITTED 4 evals for ${key} tag=${eval_tag} seed=${EVAL_SEED}"
    return 0
  fi
  log "ERROR submitting evals for ${key} (will retry)"
  return 1
}

# Olmo letter + letter_option: only eval after BOTH are settled (ready ckpt or terminal fail).
handle_olmo_pair() {
  local status_l status_o
  status_l=$(probe_one "olmo_mcq_let" "olmo3_7b_think" "${RUN_NAME_OLL}" "${TRAIN_JID_OLL}" || true)
  status_o=$(probe_one "olmo_mcq_opt" "olmo3_7b_think" "${RUN_NAME_OLO}" "${TRAIN_JID_OLO}" || true)

  local settled_l=0 settled_o=0
  case "${status_l}" in
    done|failed|ready:*) settled_l=1 ;;
  esac
  case "${status_o}" in
    done|failed|ready:*) settled_o=1 ;;
  esac

  if [[ "${settled_l}" -ne 1 || "${settled_o}" -ne 1 ]]; then
    log "WAIT olmo_pair let=${status_l}(jid=${TRAIN_JID_OLL},$(job_state "${TRAIN_JID_OLL}")) opt=${status_o}(jid=${TRAIN_JID_OLO},$(job_state "${TRAIN_JID_OLO}")) — need both settled before eval"
    return 1
  fi

  log "olmo_pair both settled (let=${status_l%%:*} opt=${status_o%%:*}); submitting evals as needed"

  local ok=0
  # letter
  if [[ "${status_l}" == done || "${status_l}" == failed ]]; then
    ok=$((ok + 1))
  elif [[ "${status_l}" == ready:* ]]; then
    if submit_ready "olmo_mcq_let" "olmo3_7b_think" "${RUN_NAME_OLL}" "${TRAIN_JID_OLL}" "olmo_7b_think" "${status_l#ready:}"; then
      ok=$((ok + 1))
    fi
  fi
  # option
  if [[ "${status_o}" == done || "${status_o}" == failed ]]; then
    ok=$((ok + 1))
  elif [[ "${status_o}" == ready:* ]]; then
    if submit_ready "olmo_mcq_opt" "olmo3_7b_think" "${RUN_NAME_OLO}" "${TRAIN_JID_OLO}" "olmo_7b_think" "${status_o#ready:}"; then
      ok=$((ok + 1))
    fi
  fi
  [[ "${ok}" -ge 2 ]]
}

log "host=$(hostname) job=${SLURM_JOB_ID:-manual} interval=${INTERVAL}s ckpt_step=${CKPT_STEP} eval_seed=${EVAL_SEED}"
log "watching:"
log "  1.7b_mcq_let jid=${TRAIN_JID_17L} run=${RUN_NAME_17L}"
log "  1.7b_mcq_opt jid=${TRAIN_JID_17O} run=${RUN_NAME_17O}"
log "  olmo_mcq_let jid=${TRAIN_JID_OLL} run=${RUN_NAME_OLL}  (gated with opt)"
log "  olmo_mcq_opt jid=${TRAIN_JID_OLO} run=${RUN_NAME_OLO}  (gated with let)"

while true; do
  done_count=0
  handle_one "1p7b_mcq_let" "qwen3_1.7b" "${RUN_NAME_17L}" "${TRAIN_JID_17L}" "1.7b" && done_count=$((done_count + 1))
  handle_one "1p7b_mcq_opt" "qwen3_1.7b" "${RUN_NAME_17O}" "${TRAIN_JID_17O}" "1.7b" && done_count=$((done_count + 1))
  if handle_olmo_pair; then
    done_count=$((done_count + 2))
  fi
  log "progress done=${done_count}/4"
  if [[ "${done_count}" -ge 4 ]]; then
    log "ALL trains handled; exiting"
    break
  fi
  sleep "${INTERVAL}"
done

log "watch finished"
exit 0
