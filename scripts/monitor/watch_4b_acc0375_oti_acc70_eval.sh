#!/bin/bash
# Watch two train jobs and submit evals when checkpoint-100 is ready:
#   3137279  qwen3-4b          snt_tnt acc0375  → 8 evals (aime24/25/26+hmmt25 × think/nothink)
#   3137343  qwen3-4b-instruct snt_tnt acc70    → 4 evals (aime24/25/26+hmmt25 nothink)
#
# Usage:
#   nohup bash scripts/monitor/watch_4b_acc0375_oti_acc70_eval.sh \
#     > log/monitor/watch_4b_acc0375_oti_acc70_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

POLL_SEC=${POLL_SEC:-120}
STEP=${STEP:-100}
BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

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

mode_suffix() {
  case "$1" in
    nothink) echo nt ;;
    think) echo th ;;
    *) echo eval ;;
  esac
}

submit_full8() {
  local train_jid="$1" ckpt="$2" eval_tag="$3" prefix="$4" eval_subdir="$5"
  local lock="${BASE_DIR}/log/monitor/eval_submitted_${train_jid}.lock"
  local jobs_file="${BASE_DIR}/log/monitor/eval_jobs_${train_jid}.txt"
  local ds mode script job_name job_id

  mkdir -p "${BASE_DIR}/log/monitor"
  if [[ -f "${lock}" ]]; then
    log "[${train_jid}] evals already submitted; skip"
    return 0
  fi
  : > "${lock}"
  : > "${jobs_file}"

  cd "${BASE_DIR}"
  log "[${train_jid}] submitting 8 evals ckpt=${ckpt} tag=${eval_tag} dir=scripts/eval/${eval_subdir}"
  for ds in aime24 aime25 aime26 hmmt25; do
    for mode in nothink think; do
      script="${BASE_DIR}/scripts/eval/${eval_subdir}/${ds}_${mode}.sh"
      [[ -f "${script}" ]] || { log "ERROR: missing ${script}"; return 1; }
      job_name="${prefix}_$(short_ds "${ds}")_$(mode_suffix "${mode}")"
      job_id=$(
        sbatch --parsable \
          --job-name="${job_name}" \
          --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${ckpt}",EVAL_TAG="${eval_tag}" \
          "${script}" 2>/dev/null | tail -1
      )
      log "[${train_jid}] submitted ${ds}_${mode}: job ${job_id} (${job_name})"
      echo "${job_id} ${job_name} ${eval_tag} scripts/eval/${eval_subdir}/${ds}_${mode}.sh" >> "${jobs_file}"
    done
  done
  log "[${train_jid}] done. job ids -> ${jobs_file}"
}

submit_instruct4() {
  # aime24/25/26 + hmmt25, nothink only
  local train_jid="$1" ckpt="$2" eval_tag="$3" prefix="$4"
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
  log "[${train_jid}] submitting 4 instruct nothink evals ckpt=${ckpt} tag=${eval_tag}"
  for ds in aime24 aime25 aime26 hmmt25; do
    script="${BASE_DIR}/scripts/eval/4b_instruct/${ds}_nothink.sh"
    [[ -f "${script}" ]] || { log "ERROR: missing ${script}"; return 1; }
    job_name="${prefix}_$(short_ds "${ds}")_nt"
    job_id=$(
      sbatch --parsable \
        --job-name="${job_name}" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${ckpt}",EVAL_TAG="${eval_tag}" \
        "${script}" 2>/dev/null | tail -1
    )
    log "[${train_jid}] submitted ${ds}_nothink: job ${job_id} (${job_name})"
    echo "${job_id} ${job_name} ${eval_tag} scripts/eval/4b_instruct/${ds}_nothink.sh" >> "${jobs_file}"
  done
  log "[${train_jid}] done. job ids -> ${jobs_file}"
}

watch_until_ckpt() {
  local train_jid="$1" ckpt="$2" waited=0
  log "[${train_jid}] watching ckpt=${ckpt}"
  while true; do
    if ckpt_ready "${ckpt}"; then
      log "[${train_jid}] checkpoint-${STEP} ready"
      return 0
    fi
    if ! job_in_queue "${train_jid}"; then
      local st
      st=$(job_state "${train_jid}")
      log "[${train_jid}] left queue; state=${st}"
      sleep 15
      if ckpt_ready "${ckpt}"; then
        log "[${train_jid}] ckpt found after job end"
        return 0
      fi
      log "[${train_jid}] ERROR: no usable checkpoint-${STEP} (${st})"
      ls -la "$(dirname "${ckpt}")" 2>/dev/null || true
      return 1
    fi
    local st_line
    st_line=$(squeue -h -j "${train_jid}" -o '%T %M %N' 2>/dev/null | head -n1 || true)
    log "[${train_jid}] waiting... [${st_line:-?}] waited=${waited}s"
    sleep "${POLL_SEC}"
    waited=$((waited + POLL_SEC))
  done
}

watch_4b_acc0375() {
  local jid=3137279
  local model_tag=qwen3_4b
  local run_name=snt_tnt_1e_6_ot_acc0375_4b
  local eval_tag=snt_tnt_acc0375_4b_ckpt100
  local prefix=snt_tnt_acc0375_4b
  local ckpt="${BASE_DIR}/outputs/${model_tag}/${run_name}/${jid}/checkpoint-${STEP}"
  watch_until_ckpt "${jid}" "${ckpt}" || return 1
  submit_full8 "${jid}" "${ckpt}" "${eval_tag}" "${prefix}" "4b"
}

watch_oti_acc70() {
  local jid=3137343
  local model_tag=qwen3_4b_instruct
  local run_name=snt_tnt_1e_6_ot_acc70_oti
  local eval_tag=snt_tnt_acc70_oti_ckpt100
  local prefix=snt_tnt_acc70_oti
  local ckpt="${BASE_DIR}/outputs/${model_tag}/${run_name}/${jid}/checkpoint-${STEP}"
  watch_until_ckpt "${jid}" "${ckpt}" || return 1
  submit_instruct4 "${jid}" "${ckpt}" "${eval_tag}" "${prefix}"
}

main() {
  mkdir -p "${BASE_DIR}/log/monitor"
  log "monitor start: 4b-acc0375→8evals + oti-acc70→4evals, poll=${POLL_SEC}s"
  local pids=() rc=0
  watch_4b_acc0375 &
  pids+=($!)
  watch_oti_acc70 &
  pids+=($!)
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  log "monitor done (rc=${rc})"
  return "${rc}"
}

main "$@"
