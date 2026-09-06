#!/bin/bash
# CPU-node watcher: when each c256+adv_t4 train finishes, submit 4 think evals.
# Manifest TSV columns: model_key model_tag run_name train_jid script
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
cd "${BASE_DIR}"

MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/jsd005_c256_adv_t4/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-60}
SUBMIT="${BASE_DIR}/scripts/train/jsd005_c256_adv_t4/submit_four_think_evals.sh"
STATE_DIR="${BASE_DIR}/log/train/jsd005_c256_adv_t4/watch_state"
LOG="${BASE_DIR}/log/train/jsd005_c256_adv_t4/watch.${SLURM_JOB_ID:-manual}.log"
mkdir -p "${STATE_DIR}" "$(dirname "${LOG}")"
chmod +x "${SUBMIT}" 2>/dev/null || true

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

job_in_queue() {
  local jid=$1
  local st
  st=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
  [[ -n "${st}" ]]
}

job_state() {
  local jid=$1
  # Prefer live queue; fall back to sacct.
  local st
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
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

find_ckpt() {
  # Prefer checkpoint-100: final/ is often incomplete when the job exits with error
  # after save_steps but before trainer.save_model().
  local model_tag=$1 run_name=$2 train_jid=$3
  local root="${BASE_DIR}/outputs/${model_tag}/${run_name}/${train_jid}"
  if ckpt_has_weights "${root}/checkpoint-100"; then
    echo "${root}/checkpoint-100"
    return 0
  fi
  if ckpt_has_weights "${root}/final"; then
    echo "${root}/final"
    return 0
  fi
  return 1
}

if [[ ! -f "${MANIFEST}" ]]; then
  log "ERROR missing manifest: ${MANIFEST}"
  exit 1
fi

mapfile -t ROWS < <(grep -vE '^\s*(#|$)' "${MANIFEST}" || true)
total=${#ROWS[@]}
if [[ "${total}" -eq 0 ]]; then
  log "ERROR empty manifest"
  exit 1
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none} interval=${INTERVAL}s"
log "manifest=${MANIFEST} rows=${total}"

while true; do
  done_count=0
  pending=0
  failed=0

  for row in "${ROWS[@]}"; do
    # TSV: model_key model_tag run_name train_jid script
    IFS=$'\t' read -r model_key model_tag run_name train_jid _script <<<"${row}"
    stamp="${STATE_DIR}/${model_key}_${train_jid}.done"
    fail_stamp="${STATE_DIR}/${model_key}_${train_jid}.failed"

    if [[ -f "${stamp}" ]]; then
      done_count=$((done_count + 1))
      continue
    fi
    # Stale .failed stamps are ignored if a usable checkpoint later appears.
    if [[ -f "${fail_stamp}" ]] && ! find_ckpt "${model_tag}" "${run_name}" "${train_jid}" >/dev/null; then
      failed=$((failed + 1))
      continue
    fi

    if job_in_queue "${train_jid}"; then
      st=$(job_state "${train_jid}")
      pending=$((pending + 1))
      log "WAIT ${model_key} jid=${train_jid} state=${st}"
      continue
    fi

    st=$(job_state "${train_jid}")

    # Even if train FAILED/TIMEOUT, submit evals when checkpoint-100 (or complete final) exists.
    if ckpt=$(find_ckpt "${model_tag}" "${run_name}" "${train_jid}"); then
      log "READY ${model_key} jid=${train_jid} -> ${ckpt} (train_state=${st})"
      sleep "${POST_DONE_WAIT}"  # allow late weight flush
      if ! ckpt=$(find_ckpt "${model_tag}" "${run_name}" "${train_jid}"); then
        pending=$((pending + 1))
        log "WAIT ${model_key} jid=${train_jid} (ckpt disappeared after wait)"
        continue
      fi
      EVAL_TAG="${run_name}_$(basename "${ckpt}")"
      if MODEL_KEY="${model_key}" CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${EVAL_TAG}" \
          BASE_DIR="${BASE_DIR}" bash "${SUBMIT}" >>"${LOG}" 2>&1; then
        date -Is >"${stamp}"
        echo "${ckpt}" >>"${stamp}"
        rm -f "${fail_stamp}"
        done_count=$((done_count + 1))
        log "SUBMITTED evals for ${model_key} tag=${EVAL_TAG}"
      else
        pending=$((pending + 1))
        log "ERROR submitting evals for ${model_key} (will retry)"
      fi
      continue
    fi

    case "${st}" in
      FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
        log "FAIL ${model_key} jid=${train_jid} state=${st} and no usable ckpt-100/final"
        date -Is >"${fail_stamp}"
        echo "${st}" >>"${fail_stamp}"
        failed=$((failed + 1))
        ;;
      *)
        pending=$((pending + 1))
        log "WAIT ${model_key} jid=${train_jid} left queue (state=${st}) but no usable ckpt yet"
        ;;
    esac
  done

  log "progress done=${done_count}/${total} pending=${pending} failed=${failed}"
  if [[ $((done_count + failed)) -ge "${total}" ]]; then
    log "ALL trains handled (done=${done_count} failed=${failed}); exiting"
    break
  fi
  sleep "${INTERVAL}"
done

log "watch finished"
exit 0
