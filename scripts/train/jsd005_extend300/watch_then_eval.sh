#!/bin/bash
# CPU-node watcher: when each 50n checkpoint under a resume run is complete,
# submit 4 think evals as one job array.
#
# Manifest TSV columns:
#   model_key  model_tag  run_name  output_jid  train_jid  eval_steps
# eval_steps e.g. 150,200,250,300
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
cd "${BASE_DIR}"

MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/jsd005_extend300/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}
SUBMIT="${BASE_DIR}/scripts/train/jsd005_extend300/submit_four_think_evals.sh"
STATE_DIR="${BASE_DIR}/log/train/jsd005_extend300/watch_state"
LOG="${BASE_DIR}/log/train/jsd005_extend300/watch.${SLURM_JOB_ID:-manual}.log"
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
  # Wait until DeepSpeed/HF save has finished writing weights (not just an empty dir).
  [[ -f "${p}/model.safetensors" ]] \
    || [[ -f "${p}/model.safetensors.index.json" ]] \
    || [[ -f "${p}/model-00001-of-00001.safetensors" ]] \
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

ckpt_complete() {
  local p=$1
  ckpt_has_weights "${p}" || return 1
  # Prefer a stable signal that the Trainer finished the checkpoint write.
  [[ -f "${p}/trainer_state.json" ]] || return 1
}

eval_tag_for() {
  local model_key=$1 step=$2
  case "${model_key}" in
    qwen3_1.7b|1.7b) echo "st_tt_clip005_1e6_ot_1p7b_ckpt${step}" ;;
    olmo3_7b_think|olmo_7b_think) echo "st_tt_clip005_1e6_olmo7bt_ckpt${step}" ;;
    *) echo "${model_key}_ckpt${step}" ;;
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

# Expand to (model_key, ckpt_path, eval_tag, train_jid) targets.
declare -a TARGETS=()
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r model_key model_tag run_name output_jid train_jid eval_steps <<<"${row}"
  IFS=',' read -r -a steps <<< "${eval_steps}"
  root="${BASE_DIR}/outputs/${model_tag}/${run_name}/${output_jid}"
  for step in "${steps[@]}"; do
    step="${step// /}"
    [[ -n "${step}" ]] || continue
    tag="$(eval_tag_for "${model_key}" "${step}")"
    TARGETS+=("${model_key}|${root}/checkpoint-${step}|${tag}|${train_jid}")
  done
done

ntarget=${#TARGETS[@]}
log "host=$(hostname) job=${SLURM_JOB_ID:-none} interval=${INTERVAL}s post_wait=${POST_DONE_WAIT}s"
log "manifest=${MANIFEST} runs=${total} targets=${ntarget}"

while true; do
  done_count=0
  pending=0
  failed=0

  for t in "${TARGETS[@]}"; do
    IFS='|' read -r model_key ckpt eval_tag train_jid <<<"${t}"
    stamp="${STATE_DIR}/${model_key}_${eval_tag}.submitted"
    fail_stamp="${STATE_DIR}/${model_key}_${eval_tag}.failed"

    if [[ -f "${stamp}" ]]; then
      done_count=$((done_count + 1))
      continue
    fi

    if [[ -f "${fail_stamp}" ]] && ! ckpt_complete "${ckpt}"; then
      failed=$((failed + 1))
      continue
    fi

    if ckpt_complete "${ckpt}"; then
      log "READY ${model_key} -> ${ckpt}"
      sleep "${POST_DONE_WAIT}"
      if ! ckpt_complete "${ckpt}"; then
        pending=$((pending + 1))
        log "WAIT ${model_key} ${ckpt} (incomplete after wait)"
        continue
      fi
      if MODEL_KEY="${model_key}" CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${eval_tag}" \
          SEED=42 BASE_DIR="${BASE_DIR}" bash "${SUBMIT}" >>"${LOG}" 2>&1; then
        done_count=$((done_count + 1))
        rm -f "${fail_stamp}"
        log "SUBMITTED evals for ${model_key} tag=${eval_tag}"
      else
        pending=$((pending + 1))
        log "ERROR submitting ${model_key} tag=${eval_tag} (will retry)"
      fi
      continue
    fi

    # No ckpt yet: if train already terminal-failed, mark this step failed.
    if ! job_in_queue "${train_jid}"; then
      st=$(job_state "${train_jid}")
      case "${st}" in
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
          log "FAIL ${model_key} tag=${eval_tag} train=${train_jid} state=${st} no usable ckpt"
          date -Is >"${fail_stamp}"
          echo "${st}" >>"${fail_stamp}"
          failed=$((failed + 1))
          continue
          ;;
      esac
    fi

    pending=$((pending + 1))
    if job_in_queue "${train_jid}"; then
      st=$(job_state "${train_jid}")
      log "WAIT ${model_key} tag=${eval_tag} train=${train_jid} state=${st}"
    else
      st=$(job_state "${train_jid}")
      log "WAIT ${model_key} tag=${eval_tag} (no ckpt yet; train_state=${st})"
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
