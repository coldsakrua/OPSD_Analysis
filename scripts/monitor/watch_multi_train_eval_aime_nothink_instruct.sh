#!/bin/bash
# Watch Instruct train jobs; when checkpoint-100 is ready (or job ends with
# ckpt present, including FAILED), submit AIME24/25/26 nothink (3 evals each)
# via scripts/eval/4b_instruct (logs -> log/eval/4b_instruct/).
#
# Usage:
#   nohup bash scripts/monitor/watch_multi_train_eval_aime_nothink_instruct.sh \
#     > log/monitor/watch_oti_hyper_2965228_33.log 2>&1 &
#
# Spec: JOB_ID|RUN_NAME|EVAL_TAG|JOB_NAME_PREFIX
# Override: SPECS_OVERRIDE='jid|run|tag|prefix ...'
set -euo pipefail

POLL_SEC=${POLL_SEC:-30}
STEP=${STEP:-100}
BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

if [[ -n "${SPECS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  SPECS=(${SPECS_OVERRIDE})
else
  SPECS=(
    "2965228|snt_tnt_lr2e6_oti|snt_oti_lr2e6_ckpt100|snt_oti_lr2e6"
    "2965229|snt_tnt_lr5e6_oti|snt_oti_lr5e6_ckpt100|snt_oti_lr5e6"
    "2965230|snt_tnt_jsd1e7_oti|snt_oti_c1e7_ckpt100|snt_oti_c1e7"
    "2965231|snt_tnt_jsd1e5_oti|snt_oti_c1e5_ckpt100|snt_oti_c1e5"
    "2965232|snt_tnt_gbs16_oti|snt_oti_g16_ckpt100|snt_oti_g16"
    "2965233|snt_tnt_gbs64_oti|snt_oti_g64_ckpt100|snt_oti_g64"
  )
fi

EVAL_SCRIPTS=(
  "${BASE_DIR}/scripts/eval/4b_instruct/aime24_nothink.sh"
  "${BASE_DIR}/scripts/eval/4b_instruct/aime25_nothink.sh"
  "${BASE_DIR}/scripts/eval/4b_instruct/aime26_nothink.sh"
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
  log "[${train_jid}] submitting 3 instruct nothink evals ckpt=${ckpt} tag=${eval_tag}"
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
  log "multi-monitor start: ${#SPECS[@]} jobs × 3 instruct nothink, poll=${POLL_SEC}s step=${STEP}"
  local pids=() spec jid run_name eval_tag prefix rc=0

  for spec in "${SPECS[@]}"; do
    [[ -n "${spec}" ]] || continue
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
