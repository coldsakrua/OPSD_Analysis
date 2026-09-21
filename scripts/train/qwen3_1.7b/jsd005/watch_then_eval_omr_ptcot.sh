#!/bin/bash
# CPU-node watcher: when Qwen3-1.7B OMR postthink+cot OPSD trains finish
# checkpoint-100, submit 4 think evals (SEED=42).
#
# Manifest TSV columns: model_key model_tag run_name train_jid script eval_tag
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}}
cd "${BASE_DIR}"

MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/qwen3_1p7b_omr_ptcot/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}
SUBMIT="${BASE_DIR}/scripts/train/qwen3_1.7b/jsd005/submit_four_think_evals_omr_ptcot.sh"
STATE_DIR="${BASE_DIR}/log/train/qwen3_1p7b_omr_ptcot/watch_state"
LOG="${BASE_DIR}/log/train/qwen3_1p7b_omr_ptcot/watch.${SLURM_JOB_ID:-manual}.log"
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
  if ckpt_complete "${root}/checkpoint-100"; then
    echo "${root}/checkpoint-100"
    return 0
  fi
  if ckpt_complete "${root}/final"; then
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

log "host=$(hostname) job=${SLURM_JOB_ID:-none} interval=${INTERVAL}s post_wait=${POST_DONE_WAIT}s"
log "manifest=${MANIFEST} rows=${total}"

while true; do
  done_count=0
  pending=0
  failed=0

  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r model_key model_tag run_name train_jid _script eval_tag <<<"${row}"
    stamp="${STATE_DIR}/${eval_tag}_${train_jid}.done"
    fail_stamp="${STATE_DIR}/${eval_tag}_${train_jid}.failed"

    if [[ -f "${stamp}" ]]; then
      done_count=$((done_count + 1))
      continue
    fi
    if [[ -f "${fail_stamp}" ]] && ! find_ckpt "${model_tag}" "${run_name}" "${train_jid}" >/dev/null; then
      failed=$((failed + 1))
      continue
    fi

    if ckpt=$(find_ckpt "${model_tag}" "${run_name}" "${train_jid}"); then
      st=$(job_state "${train_jid}")
      log "READY ${eval_tag} jid=${train_jid} -> ${ckpt} (train_state=${st})"
      sleep "${POST_DONE_WAIT}"
      if ! ckpt=$(find_ckpt "${model_tag}" "${run_name}" "${train_jid}"); then
        pending=$((pending + 1))
        log "WAIT ${eval_tag} jid=${train_jid} (ckpt disappeared after wait)"
        continue
      fi
      ckpt_base="$(basename "${ckpt}")"
      submit_tag="${eval_tag}"
      if [[ "${ckpt_base}" != checkpoint-100 ]]; then
        submit_tag="${eval_tag%_ckpt100}_${ckpt_base}"
      fi
      if MODEL_KEY="${model_key}" CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${submit_tag}" \
          SEED=42 BASE_DIR="${BASE_DIR}" bash "${SUBMIT}" >>"${LOG}" 2>&1; then
        date -Is >"${stamp}"
        echo "${ckpt}" >>"${stamp}"
        rm -f "${fail_stamp}"
        done_count=$((done_count + 1))
        log "SUBMITTED evals for ${eval_tag} tag=${submit_tag}"
      else
        pending=$((pending + 1))
        log "ERROR submitting evals for ${eval_tag} (will retry)"
      fi
      continue
    fi

    if job_in_queue "${train_jid}"; then
      st=$(job_state "${train_jid}")
      pending=$((pending + 1))
      log "WAIT ${eval_tag} jid=${train_jid} state=${st}"
      continue
    fi

    st=$(job_state "${train_jid}")
    case "${st}" in
      FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
        log "FAIL ${eval_tag} jid=${train_jid} state=${st} and no usable ckpt-100/final"
        date -Is >"${fail_stamp}"
        echo "${st}" >>"${fail_stamp}"
        failed=$((failed + 1))
        ;;
      *)
        pending=$((pending + 1))
        log "WAIT ${eval_tag} jid=${train_jid} left queue (state=${st}) but no usable ckpt yet"
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
