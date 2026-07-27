#!/bin/bash
# Watch one or more train jobs; when checkpoint-100 is ready (or job ends with
# ckpt present), submit AIME24/25/26 nothink evals.
#
# Usage (multi):
#   nohup bash scripts/monitor/watch_multi_train_eval_aime_nothink.sh \
#     > log/monitor/watch_withoutgt_2958636_38.log 2>&1 &
#
# Spec lines: JOB_ID|RUN_NAME|EVAL_TAG|JOB_NAME_PREFIX
set -euo pipefail

POLL_SEC=${POLL_SEC:-30}
STEP=${STEP:-100}
BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

# Default: the three withoutgt runs currently in queue.
SPECS=(
  "${SPECS_OVERRIDE:-}"
)
if [[ -z "${SPECS[0]:-}" ]]; then
  SPECS=(
    "2958636|snt_tt_same_ot_1e_6|snt_tt_same_ckpt100|snt_tt_same"
    "2958637|snt_tt_encourage_ot_1e_6|snt_tt_enc_ckpt100|snt_tt_enc"
    "2958638|snt_tt_irrelevant_ot_1e_6|snt_tt_irr_ckpt100|snt_tt_irr"
  )
fi

EVAL_SCRIPTS=(
  "${BASE_DIR}/scripts/eval/4b/aime24_nothink.sh"
  "${BASE_DIR}/scripts/eval/4b/aime25_nothink.sh"
  "${BASE_DIR}/scripts/eval/4b/aime26_nothink.sh"
)

log() {
  echo "[$(date '+%F %T')] $*"
}

job_in_queue() {
  local jid="$1"
  squeue -h -j "${jid}" 2>/dev/null | grep -q .
}

job_state() {
  local jid="$1" st
  st=$(sacct -j "${jid}" -n -X -o State --parsable2 2>/dev/null | head -n1 | tr -d '[:space:]')
  if [[ -n "${st}" ]]; then
    echo "${st}"
    return
  fi
  st=$(squeue -h -j "${jid}" -o '%T' 2>/dev/null | head -n1 | tr -d '[:space:]')
  echo "${st:-UNKNOWN}"
}

ckpt_ready() {
  local ckpt="$1"
  [[ -d "${ckpt}" ]] || return 1
  [[ -f "${ckpt}/config.json" ]] || return 1
  [[ -f "${ckpt}/model.safetensors.index.json" || -f "${ckpt}/model.safetensors" || -f "${ckpt}/pytorch_model.bin" ]] || return 1
  [[ -f "${ckpt}/tokenizer.json" || -f "${ckpt}/tokenizer_config.json" ]] || return 1
  return 0
}

submit_evals() {
  local train_jid="$1" ckpt="$2" eval_tag="$3" prefix="$4"
  local lock="${BASE_DIR}/log/monitor/eval_submitted_${train_jid}.lock"
  local jobs_file="${BASE_DIR}/log/monitor/eval_jobs_${train_jid}.txt"
  local script ds job_name job_id

  mkdir -p "${BASE_DIR}/log/monitor"
  if [[ -f "${lock}" ]]; then
    log "[${train_jid}] evals already submitted; skip"
    return 0
  fi
  : > "${lock}"
  : > "${jobs_file}"

  cd "${BASE_DIR}"
  log "[${train_jid}] submitting 3 nothink evals ckpt=${ckpt} tag=${eval_tag}"
  for script in "${EVAL_SCRIPTS[@]}"; do
    [[ -f "${script}" ]] || { log "ERROR: missing ${script}"; return 1; }
    ds=$(basename "${script}" | sed -E 's/_nothink\.sh$//')
    job_name="${prefix}_${ds/aime/a}_nt"
    job_id=$(
      sbatch --parsable \
        --job-name="${job_name}" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${ckpt}",EVAL_TAG="${eval_tag}" \
        "${script}" 2>/dev/null | tail -1
    )
    log "[${train_jid}] submitted ${ds}: job ${job_id} (${job_name})"
    echo "${job_id} ${job_name}" >> "${jobs_file}"
  done
}

watch_one() {
  local train_jid="$1" run_name="$2" eval_tag="$3" prefix="$4"
  local output_dir="${BASE_DIR}/outputs/${run_name}/${train_jid}"
  local ckpt="${output_dir}/checkpoint-${STEP}"
  local waited=0

  log "[${train_jid}] watch start run=${run_name} ckpt=${ckpt} tag=${eval_tag}"

  while true; do
    if ckpt_ready "${ckpt}"; then
      log "[${train_jid}] checkpoint-${STEP} ready"
      submit_evals "${train_jid}" "${ckpt}" "${eval_tag}" "${prefix}"
      return 0
    fi

    if ! job_in_queue "${train_jid}"; then
      local st
      st=$(job_state "${train_jid}")
      log "[${train_jid}] left queue; state=${st}"
      sleep 10
      if ckpt_ready "${ckpt}"; then
        log "[${train_jid}] ckpt found after job end; submitting"
        submit_evals "${train_jid}" "${ckpt}" "${eval_tag}" "${prefix}"
        return 0
      fi
      log "[${train_jid}] ERROR: no usable checkpoint-${STEP} (${st})"
      ls -la "${output_dir}" 2>/dev/null || true
      return 1
    fi

    local st_line
    st_line=$(squeue -h -j "${train_jid}" -o '%T %M %N' 2>/dev/null | head -n1 || true)
    log "[${train_jid}] waiting... [${st_line:-?}] waited=${waited}s"
    sleep "${POLL_SEC}"
    waited=$((waited + POLL_SEC))
  done
}

main() {
  mkdir -p "${BASE_DIR}/log/monitor"
  log "multi-monitor start: ${#SPECS[@]} jobs, poll=${POLL_SEC}s"
  local pids=() spec jid run_name eval_tag prefix rc=0

  for spec in "${SPECS[@]}"; do
    IFS='|' read -r jid run_name eval_tag prefix <<<"${spec}"
    watch_one "${jid}" "${run_name}" "${eval_tag}" "${prefix}" &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  log "multi-monitor done (rc=${rc})"
  return "${rc}"
}

main "$@"
