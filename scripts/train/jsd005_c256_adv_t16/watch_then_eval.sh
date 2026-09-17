#!/bin/bash
# CPU-node watcher: wait until ALL c256+adv_t16 trains finish, then submit
# think evals for every usable checkpoint.
# GPU rule: 16-GPU association, keep >=4 GPUs free (submit at most 12 at a time).
# Manifest TSV columns: model_key model_tag run_name train_jid script
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
cd "${BASE_DIR}"

MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/jsd005_c256_adv_t16/submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-60}
EVALS_PER_MODEL=${EVALS_PER_MODEL:-4}
SUBMIT="${BASE_DIR}/scripts/train/jsd005_c256_adv_t16/submit_four_think_evals.sh"
STATE_DIR="${BASE_DIR}/log/train/jsd005_c256_adv_t16/watch_state"
LOG="${BASE_DIR}/log/train/jsd005_c256_adv_t16/watch.${SLURM_JOB_ID:-manual}.log"
BATCH_STAMP="${STATE_DIR}/all_evals.submitted"
mkdir -p "${STATE_DIR}" "$(dirname "${LOG}")"
chmod +x "${SUBMIT}" 2>/dev/null || true
# shellcheck source=/dev/null
source "${BASE_DIR}/scripts/train/common/gpu_quota.sh"

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
log "mode=wait-all-then-submit; gpu_quota=${GPU_QUOTA} reserve=${GPU_RESERVE} cap=$(gpu_cap) in_use=$(user_gpu_in_use)"

if [[ -f "${BATCH_STAMP}" ]]; then
  log "SKIP all evals already submitted ($(cat "${BATCH_STAMP}"))"
  exit 0
fi

while true; do
  pending=0
  ready=0
  failed=0
  declare -a READY_KEYS=()
  declare -a READY_CKPTS=()
  declare -a READY_TAGS=()

  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r model_key model_tag run_name train_jid _script <<<"${row}"

    if job_in_queue "${train_jid}"; then
      st=$(job_state "${train_jid}")
      pending=$((pending + 1))
      log "WAIT ${model_key} jid=${train_jid} state=${st}"
      continue
    fi

    st=$(job_state "${train_jid}")
    if ckpt=$(find_ckpt "${model_tag}" "${run_name}" "${train_jid}"); then
      ready=$((ready + 1))
      READY_KEYS+=("${model_key}")
      READY_CKPTS+=("${ckpt}")
      READY_TAGS+=("${run_name}_$(basename "${ckpt}")")
      log "READY ${model_key} jid=${train_jid} -> ${ckpt} (train_state=${st})"
      continue
    fi

    case "${st}" in
      FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
        failed=$((failed + 1))
        log "FAIL ${model_key} jid=${train_jid} state=${st} and no usable ckpt-100/final"
        ;;
      *)
        pending=$((pending + 1))
        log "WAIT ${model_key} jid=${train_jid} left queue (state=${st}) but no usable ckpt yet"
        ;;
    esac
  done

  log "progress ready=${ready} pending=${pending} failed=${failed} total=${total}"
  if [[ $((ready + failed)) -ge "${total}" && "${pending}" -eq 0 ]]; then
    log "ALL trains settled (ready=${ready} failed=${failed}); submitting evals with gpu cap=$(gpu_cap)"
    break
  fi
  sleep "${INTERVAL}"
done

if [[ "${ready}" -eq 0 ]]; then
  log "ERROR no usable checkpoints; not submitting evals"
  exit 1
fi

log "post-done wait ${POST_DONE_WAIT}s for late weight flush"
sleep "${POST_DONE_WAIT}"

submit_ok=0
submit_err=0
for i in "${!READY_KEYS[@]}"; do
  model_key="${READY_KEYS[$i]}"
  ckpt="${READY_CKPTS[$i]}"
  eval_tag="${READY_TAGS[$i]}"
  if [[ -f "${STATE_DIR}/${model_key}_${eval_tag}.submitted" ]]; then
    submit_ok=$((submit_ok + 1))
    log "SKIP ${model_key} already submitted tag=${eval_tag}"
    continue
  fi
  if ! ckpt_has_weights "${ckpt}"; then
    submit_err=$((submit_err + 1))
    log "ERROR ckpt disappeared before submit: ${model_key} ${ckpt}"
    continue
  fi
  log "HOLD-FOR-GPU ${model_key} need=${EVALS_PER_MODEL} in_use=$(user_gpu_in_use) cap=$(gpu_cap)"
  wait_for_gpu_slots "${EVALS_PER_MODEL}" | tee -a "${LOG}"
  log "SUBMIT ${model_key} tag=${eval_tag} ckpt=${ckpt}"
  if MODEL_KEY="${model_key}" CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${eval_tag}" \
      BASE_DIR="${BASE_DIR}" bash "${SUBMIT}" >>"${LOG}" 2>&1; then
    submit_ok=$((submit_ok + 1))
    date -Is >"${STATE_DIR}/${model_key}_batch.done"
    echo "${ckpt}" >>"${STATE_DIR}/${model_key}_batch.done"
    log "SUBMITTED evals for ${model_key} tag=${eval_tag} in_use=$(user_gpu_in_use)"
  else
    submit_err=$((submit_err + 1))
    log "ERROR submitting evals for ${model_key}"
  fi
done

{
  date -Is
  echo "ok=${submit_ok} err=${submit_err}"
} >"${BATCH_STAMP}"
log "batch submit finished ok=${submit_ok} err=${submit_err} stamp=${BATCH_STAMP}"
log "watch finished"
exit 0
