#!/bin/bash
# CPU watcher: for each train row, submit code evals as each listed checkpoint appears.
#
# Manifest TSV columns:
#   model_key  model_tag  run_name  output_jid  train_jid  eval_steps
# eval_steps e.g. 150,200,250,300  (comma-separated absolute global steps)
#
# Checkpoints live under:
#   outputs/{model_tag}/{run_name}/{output_jid}/checkpoint-{step}
# (output_jid is the original run dir; train_jid is the resume Slurm job.)
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
cd "${BASE_DIR}"

MANIFEST=${MANIFEST:-${BASE_DIR}/log/train_code/ocr_extend300/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}
SUBMIT="${BASE_DIR}/scripts/train_code/ocr/submit_three_code_evals.sh"
STATE_DIR="${BASE_DIR}/log/train_code/ocr_extend300/watch_state"
LOG="${BASE_DIR}/log/train_code/ocr_extend300/watch.${SLURM_JOB_ID:-manual}.log"
mkdir -p "${STATE_DIR}" "$(dirname "${LOG}")"
chmod +x "${SUBMIT}" 2>/dev/null || true

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

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

# Expand (row, step) tasks.
declare -a TASKS=()
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r model_key model_tag run_name output_jid train_jid eval_steps <<<"${row}"
  if [[ -z "${eval_steps:-}" ]]; then
    log "ERROR row missing eval_steps: ${row}"
    exit 1
  fi
  IFS=',' read -ra steps <<<"${eval_steps}"
  for step in "${steps[@]}"; do
    step="${step//[[:space:]]/}"
    [[ -n "${step}" ]] || continue
    TASKS+=("${model_key}|${model_tag}|${run_name}|${output_jid}|${train_jid}|${step}")
  done
done

ntasks=${#TASKS[@]}
log "host=$(hostname) job=${SLURM_JOB_ID:-none} interval=${INTERVAL}s"
log "manifest=${MANIFEST} rows=${total} tasks=${ntasks}"

while true; do
  done_count=0
  pending=0
  failed=0

  for task in "${TASKS[@]}"; do
    IFS='|' read -r model_key model_tag run_name output_jid train_jid step <<<"${task}"
    eval_tag="ocr_${model_key}_ckpt${step}"
    stamp="${STATE_DIR}/${eval_tag}_${output_jid}.done"
    fail_stamp="${STATE_DIR}/${eval_tag}_${output_jid}.failed"
    root="${BASE_DIR}/outputs/${model_tag}/${run_name}/${output_jid}"
    ckpt_path="${root}/checkpoint-${step}"

    if [[ -f "${stamp}" ]]; then
      done_count=$((done_count + 1))
      continue
    fi
    if [[ -f "${fail_stamp}" ]] && ! ckpt_complete "${ckpt_path}"; then
      failed=$((failed + 1))
      continue
    fi

    if ckpt_complete "${ckpt_path}"; then
      st=$(job_state "${train_jid}")
      log "READY ${eval_tag} output=${output_jid} train=${train_jid} -> ${ckpt_path} (train_state=${st})"
      sleep "${POST_DONE_WAIT}"
      if ! ckpt_complete "${ckpt_path}"; then
        pending=$((pending + 1))
        log "WAIT ${eval_tag} (ckpt incomplete after wait)"
        continue
      fi
      if MODEL_KEY="${model_key}" CHECKPOINT_PATH="${ckpt_path}" EVAL_TAG="${eval_tag}" \
          SEED=42 AUTO_JUDGE=1 BASE_DIR="${BASE_DIR}" bash "${SUBMIT}" >>"${LOG}" 2>&1; then
        date -Is >"${stamp}"
        echo "${ckpt_path}" >>"${stamp}"
        rm -f "${fail_stamp}"
        done_count=$((done_count + 1))
        log "SUBMITTED generate evals for ${eval_tag}"
      else
        pending=$((pending + 1))
        log "ERROR submitting evals for ${eval_tag} (will retry)"
      fi
      continue
    fi

    if job_in_queue "${train_jid}"; then
      pending=$((pending + 1))
      log "WAIT ${eval_tag} train=${train_jid} state=$(job_state "${train_jid}")"
      continue
    fi

    st=$(job_state "${train_jid}")
    case "${st}" in
      FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
        log "FAIL ${eval_tag} train=${train_jid} state=${st} and no usable ckpt-${step}"
        date -Is >"${fail_stamp}"
        echo "${st}" >>"${fail_stamp}"
        failed=$((failed + 1))
        ;;
      COMPLETED|COMPLETING)
        # Train finished but this step may still be flushing; keep waiting briefly.
        pending=$((pending + 1))
        log "WAIT ${eval_tag} train left queue (state=${st}) but ckpt-${step} not ready"
        ;;
      *)
        pending=$((pending + 1))
        log "WAIT ${eval_tag} train=${train_jid} state=${st} no ckpt yet"
        ;;
    esac
  done

  log "progress done=${done_count}/${ntasks} pending=${pending} failed=${failed}"
  if [[ $((done_count + failed)) -ge "${ntasks}" ]]; then
    log "ALL tasks handled (done=${done_count} failed=${failed}); exiting"
    break
  fi
  sleep "${INTERVAL}"
done

log "watch finished"
exit 0
