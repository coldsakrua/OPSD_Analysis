#!/bin/bash
# CPU watcher: when OPSD checkpoint-100 is complete, submit BOTH:
#   1) SFT checkpoint-10 evals
#   2) OPSD checkpoint-100 evals
#
# Manifest TSV columns:
#   model_key  model_tag  sft_run_name  opsd_run_name  train_jid  sft_step  opsd_step
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
cd "${BASE_DIR}"

MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/sft10_then_opsd/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}
SUBMIT="${BASE_DIR}/scripts/train/sft10_then_opsd/submit_four_think_evals.sh"
STATE_DIR="${BASE_DIR}/log/train/sft10_then_opsd/watch_state"
LOG="${BASE_DIR}/log/train/sft10_then_opsd/watch.${SLURM_JOB_ID:-manual}.log"
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

eval_tag_for() {
  local model_key=$1 kind=$2 step=$3
  case "${model_key}" in
    qwen3_1.7b|1.7b) echo "sft10_same_1p7b_${kind}_ckpt${step}" ;;
    olmo3_7b_think|olmo_7b_think) echo "sft10_same_olmo7bt_${kind}_ckpt${step}" ;;
    qwen3_4b_thinking|4b_thinking) echo "sft10_same_4bt_${kind}_ckpt${step}" ;;
    *) echo "${model_key}_${kind}_ckpt${step}" ;;
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

# Expand to targets: after opsd_step is ready, submit (sft_step + opsd_step).
# Each target is one eval submit: model_key|ckpt|eval_tag|kind|train_jid|opsd_ckpt_gate
declare -a TARGETS=()
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r model_key model_tag sft_run opsd_run train_jid sft_step opsd_step <<<"${row}"
  sft_root="${BASE_DIR}/outputs/${model_tag}/${sft_run}/${train_jid}"
  opsd_root="${BASE_DIR}/outputs/${model_tag}/${opsd_run}/${train_jid}"
  sft_ckpt="${sft_root}/checkpoint-${sft_step}"
  opsd_ckpt="${opsd_root}/checkpoint-${opsd_step}"
  sft_tag="$(eval_tag_for "${model_key}" "sft" "${sft_step}")"
  opsd_tag="$(eval_tag_for "${model_key}" "opsd" "${opsd_step}")"
  # Gate both evals on OPSD checkpoint completion (per user request).
  TARGETS+=("${model_key}|${sft_ckpt}|${sft_tag}|sft10|${train_jid}|${opsd_ckpt}")
  TARGETS+=("${model_key}|${opsd_ckpt}|${opsd_tag}|opsd100|${train_jid}|${opsd_ckpt}")
done

ntarget=${#TARGETS[@]}
log "host=$(hostname) job=${SLURM_JOB_ID:-none} interval=${INTERVAL}s post_wait=${POST_DONE_WAIT}s"
log "manifest=${MANIFEST} runs=${total} targets=${ntarget}"

while true; do
  done_count=0
  pending=0
  failed=0

  for t in "${TARGETS[@]}"; do
    IFS='|' read -r model_key ckpt eval_tag kind train_jid gate_ckpt <<<"${t}"
    stamp="${STATE_DIR}/${model_key}_${kind}_${eval_tag}.submitted"
    fail_stamp="${STATE_DIR}/${model_key}_${kind}_${eval_tag}.failed"

    if [[ -f "${stamp}" ]]; then
      done_count=$((done_count + 1))
      continue
    fi

    if [[ -f "${fail_stamp}" ]] && ! ckpt_complete "${gate_ckpt}"; then
      failed=$((failed + 1))
      continue
    fi

    # Wait until OPSD gate ckpt is complete, then also require the eval target ckpt.
    if ckpt_complete "${gate_ckpt}" && ckpt_complete "${ckpt}"; then
      log "READY ${model_key} kind=${kind} -> ${ckpt} (gate=${gate_ckpt})"
      sleep "${POST_DONE_WAIT}"
      if ! ckpt_complete "${gate_ckpt}" || ! ckpt_complete "${ckpt}"; then
        pending=$((pending + 1))
        log "WAIT ${model_key} ${kind} (incomplete after wait)"
        continue
      fi
      if MODEL_KEY="${model_key}" CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${eval_tag}" KIND="${kind}" \
          SEED=42 BASE_DIR="${BASE_DIR}" bash "${SUBMIT}" >>"${LOG}" 2>&1; then
        done_count=$((done_count + 1))
        rm -f "${fail_stamp}"
        log "SUBMITTED evals for ${model_key} kind=${kind} tag=${eval_tag}"
      else
        pending=$((pending + 1))
        log "ERROR submitting ${model_key} kind=${kind} (will retry)"
      fi
      continue
    fi

    if ! job_in_queue "${train_jid}"; then
      st=$(job_state "${train_jid}")
      case "${st}" in
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
          if ! ckpt_complete "${gate_ckpt}"; then
            log "FAIL ${model_key} kind=${kind} train=${train_jid} state=${st} no usable opsd gate ckpt"
            date -Is >"${fail_stamp}"
            echo "${st}" >>"${fail_stamp}"
            failed=$((failed + 1))
            continue
          fi
          ;;
      esac
    fi

    pending=$((pending + 1))
    if job_in_queue "${train_jid}"; then
      st=$(job_state "${train_jid}")
      log "WAIT ${model_key} kind=${kind} train=${train_jid} state=${st}"
    else
      st=$(job_state "${train_jid}")
      log "WAIT ${model_key} kind=${kind} (gate/ckpt not ready; train_state=${st})"
    fi
  done

  log "progress done=${done_count}/${ntarget} pending=${pending} failed=${failed}"
  if [[ $((done_count + failed)) -ge "${ntarget}" ]]; then
    log "ALL targets handled (done=${done_count} failed=${failed}); exiting"
    break
  fi
  sleep "${INTERVAL}"
done

log "watch finished"
exit 0
