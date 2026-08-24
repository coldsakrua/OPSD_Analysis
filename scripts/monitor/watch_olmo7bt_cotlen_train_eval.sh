#!/bin/bash
# Watch Olmo-3-7B-Think OT cotlen easy/hard train jobs.
# When checkpoint-100 is ready, submit 4× sgl evals (aime24/25/26 + hmmt25).
#
# Usage (after sbatch):
#   SPECS_OVERRIDE='jid|olmo3_7b_think|run_name|eval_tag|prefix|sgl_olmo7bt ...' \
#     nohup bash scripts/monitor/watch_olmo7bt_cotlen_train_eval.sh \
#     > log/monitor/watch_olmo7bt_cotlen_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Spec: JOB_ID|MODEL_TAG|RUN_NAME|EVAL_TAG|JOB_NAME_PREFIX|EVAL_KIND
# EVAL_KIND: sgl_olmo7bt
# ckpt: outputs/${MODEL_TAG}/${RUN_NAME}/${JOB_ID}/checkpoint-${STEP}
set -euo pipefail

POLL_SEC=${POLL_SEC:-120}
STEP=${STEP:-100}
BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

if [[ -n "${SPECS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  SPECS=(${SPECS_OVERRIDE})
else
  SPECS=()
fi

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

short_ds() {
  case "$1" in
    aime24) echo a24 ;;
    aime25) echo a25 ;;
    aime26) echo a26 ;;
    hmmt25) echo h25 ;;
    *) echo "$1" ;;
  esac
}

_submit_loop() {
  local train_jid="$1" ckpt="$2" eval_tag="$3" prefix="$4" eval_subdir="$5" mode_suffix="$6" script_suffix="$7"
  local lock="${BASE_DIR}/log/monitor/eval_submitted_${train_jid}.lock"
  local jobs_file="${BASE_DIR}/log/monitor/eval_jobs_${train_jid}.txt"
  local ds script job_name job_id

  mkdir -p "${BASE_DIR}/log/monitor"
  if [[ -f "${lock}" ]]; then
    log "[${train_jid}] evals already submitted; skip"
    return 0
  fi
  : > "${lock}"
  : > "${jobs_file}"

  cd "${BASE_DIR}"
  log "[${train_jid}] submitting 4 evals (${eval_subdir}/*_${script_suffix}.sh) ckpt=${ckpt} tag=${eval_tag}"
  for ds in aime24 aime25 aime26 hmmt25; do
    script="${BASE_DIR}/scripts/eval/${eval_subdir}/${ds}_${script_suffix}.sh"
    [[ -f "${script}" ]] || { log "ERROR: missing ${script}"; rm -f "${lock}"; return 1; }
    job_name="${prefix}_$(short_ds "${ds}")_${mode_suffix}"
    job_id=$(
      sbatch --parsable \
        --job-name="${job_name}" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${ckpt}",EVAL_TAG="${eval_tag}" \
        "${script}" 2>/dev/null | tail -1
    )
    if [[ -z "${job_id}" ]]; then
      log "ERROR: sbatch failed for ${script}"
      rm -f "${lock}"
      return 1
    fi
    log "[${train_jid}] submitted ${ds}_${script_suffix}: job ${job_id} (${job_name})"
    echo "${job_id} ${job_name} ${eval_tag} scripts/eval/${eval_subdir}/${ds}_${script_suffix}.sh" >> "${jobs_file}"
  done
  log "[${train_jid}] done. job ids -> ${jobs_file}"
}

submit_for_kind() {
  local kind="$1" train_jid="$2" ckpt="$3" eval_tag="$4" prefix="$5"
  case "${kind}" in
    sgl_olmo7bt)
      _submit_loop "${train_jid}" "${ckpt}" "${eval_tag}" "${prefix}" "olmo_7b_think" "sgl" "sgl"
      ;;
    *)
      log "ERROR: unknown EVAL_KIND=${kind}"
      return 1
      ;;
  esac
}

watch_one() {
  local train_jid="$1" model_tag="$2" run_name="$3" eval_tag="$4" prefix="$5" kind="$6"
  local output_dir="${BASE_DIR}/outputs/${model_tag}/${run_name}/${train_jid}"
  local ckpt="${output_dir}/checkpoint-${STEP}"
  local waited=0

  log "[${train_jid}] watch start model=${model_tag} run=${run_name} kind=${kind} ckpt=${ckpt}"

  while true; do
    if ckpt_ready "${ckpt}"; then
      log "[${train_jid}] checkpoint-${STEP} ready"
      submit_for_kind "${kind}" "${train_jid}" "${ckpt}" "${eval_tag}" "${prefix}"
      return 0
    fi

    if ! job_in_queue "${train_jid}"; then
      local st
      st=$(job_state "${train_jid}")
      log "[${train_jid}] left queue; state=${st}"
      sleep 15
      if ckpt_ready "${ckpt}"; then
        log "[${train_jid}] ckpt found after job end; submitting"
        submit_for_kind "${kind}" "${train_jid}" "${ckpt}" "${eval_tag}" "${prefix}"
        return 0
      fi
      local final_ckpt="${output_dir}/final"
      if ckpt_ready "${final_ckpt}"; then
        log "[${train_jid}] using final/ instead of checkpoint-${STEP}"
        submit_for_kind "${kind}" "${train_jid}" "${final_ckpt}" "${eval_tag}" "${prefix}"
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
  if [[ ${#SPECS[@]} -eq 0 ]]; then
    log "ERROR: no SPECS; set SPECS_OVERRIDE with train job ids"
    exit 1
  fi
  log "olmo7bt cotlen monitor start: ${#SPECS[@]} trains × 4 sgl evals, poll=${POLL_SEC}s step=${STEP}"
  local pids=() spec jid model_tag run_name eval_tag prefix kind rc=0

  for spec in "${SPECS[@]}"; do
    [[ -n "${spec}" ]] || continue
    IFS='|' read -r jid model_tag run_name eval_tag prefix kind <<<"${spec}"
    watch_one "${jid}" "${model_tag}" "${run_name}" "${eval_tag}" "${prefix}" "${kind}" &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  log "olmo7bt cotlen monitor done (rc=${rc})"
  return "${rc}"
}

main "$@"
