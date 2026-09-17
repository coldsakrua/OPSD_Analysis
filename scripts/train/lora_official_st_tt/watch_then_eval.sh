#!/bin/bash
# CPU-node watcher: when each official-OPSD LoRA train has a complete checkpoint-100,
# submit 4 think evals. Does NOT wait for train to finish / final/.
#
# Manifest TSV columns: model_key  model_tag  run_name  train_jid  script
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
cd "${BASE_DIR}"

MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/lora_official_st_tt/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_READY_WAIT=${POST_READY_WAIT:-90}
SUBMIT="${BASE_DIR}/scripts/train/lora_official_st_tt/submit_four_evals.sh"
STATE_DIR="${BASE_DIR}/log/train/lora_official_st_tt/watch_state"
LOG="${BASE_DIR}/log/train/lora_official_st_tt/watch.${SLURM_JOB_ID:-manual}.log"
mkdir -p "${STATE_DIR}" "$(dirname "${LOG}")"
chmod +x "${SUBMIT}" \
  "${BASE_DIR}/scripts/train/beta_opsd/merge_olmo_lora.sh" 2>/dev/null || true

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

job_in_queue() {
  local jid=$1
  local st
  st=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
  [[ -n "${st}" ]]
}

job_state() {
  local jid=$1
  local st
  st=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
  if [[ -n "${st}" ]]; then
    echo "${st}"
    return 0
  fi
  sacct -j "${jid}" -n -X -o State -P 2>/dev/null | head -1 | tr -d ' ' || echo UNKNOWN
}

adapter_ok() {
  local p=$1
  [[ -f "${p}/adapter_config.json" ]] || return 1
  [[ -f "${p}/adapter_model.safetensors" ]] || [[ -f "${p}/adapter_model.bin" ]]
}

ckpt100_complete() {
  local p=$1
  adapter_ok "${p}" || return 1
  [[ -f "${p}/trainer_state.json" ]]
}

find_ckpt100() {
  local model_tag=$1 run_name=$2 train_jid=$3
  local p="${BASE_DIR}/outputs/${model_tag}/${run_name}/${train_jid}/checkpoint-100"
  if ckpt100_complete "${p}"; then
    echo "${p}"
    return 0
  fi
  return 1
}

eval_tag_for() {
  local model_key=$1
  case "${model_key}" in
    qwen3_4b_thinking) echo "st_tt_lora_4bt_ckpt100" ;;
    olmo3_7b_think) echo "st_tt_lora_olmo7bt_ckpt100" ;;
    *) echo "${model_key}_lora_ckpt100" ;;
  esac
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
log "trigger=checkpoint-100 adapter (do not wait for final/)"

while true; do
  done_count=0
  pending=0
  failed=0

  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r model_key model_tag run_name train_jid _script <<<"${row}"
    stamp="${STATE_DIR}/${model_key}_${train_jid}.done"
    fail_stamp="${STATE_DIR}/${model_key}_${train_jid}.failed"

    if [[ -f "${stamp}" ]]; then
      done_count=$((done_count + 1))
      continue
    fi
    if [[ -f "${fail_stamp}" ]] && ! find_ckpt100 "${model_tag}" "${run_name}" "${train_jid}" >/dev/null; then
      failed=$((failed + 1))
      continue
    fi

    if ckpt=$(find_ckpt100 "${model_tag}" "${run_name}" "${train_jid}"); then
      log "READY ${model_key} jid=${train_jid} -> ${ckpt}"
      sleep "${POST_READY_WAIT}"
      if ! ckpt=$(find_ckpt100 "${model_tag}" "${run_name}" "${train_jid}"); then
        pending=$((pending + 1))
        log "WAIT ${model_key} jid=${train_jid} (ckpt-100 disappeared after wait)"
        continue
      fi
      EVAL_TAG="$(eval_tag_for "${model_key}")"
      if MODEL_KEY="${model_key}" CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${EVAL_TAG}" \
          bash "${SUBMIT}" >>"${LOG}" 2>&1; then
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

    if job_in_queue "${train_jid}"; then
      st=$(job_state "${train_jid}")
      pending=$((pending + 1))
      log "WAIT ${model_key} jid=${train_jid} state=${st} (no complete ckpt-100 yet)"
      continue
    fi

    st=$(job_state "${train_jid}")
    case "${st}" in
      FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
        log "FAIL ${model_key} jid=${train_jid} state=${st} and no usable ckpt-100"
        date -Is >"${fail_stamp}"
        echo "${st}" >>"${fail_stamp}"
        failed=$((failed + 1))
        ;;
      *)
        pending=$((pending + 1))
        log "WAIT ${model_key} jid=${train_jid} left queue (state=${st}) but no usable ckpt-100 yet"
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
