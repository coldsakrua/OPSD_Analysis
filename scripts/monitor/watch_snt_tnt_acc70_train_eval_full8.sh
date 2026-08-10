#!/bin/bash
# Watch snt_tnt OpenThoughts acc>=70% train jobs; when checkpoint-100 is ready,
# submit 8 evals: aime24/25/26 + hmmt25 × (think + nothink).
#
# Default targets:
#   3122844  qwen3-1.7b  snt_tnt_1e_6_ot_acc70_1p7b
#   3122883  qwen3-4b    snt_tnt_1e_6_ot_acc70_4b
#
# Usage:
#   nohup bash scripts/monitor/watch_snt_tnt_acc70_train_eval_full8.sh \
#     > log/monitor/watch_snt_tnt_acc70_full8_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Override:
#   SPECS_OVERRIDE='jid|model_tag|run_name|eval_tag|prefix|eval_subdir ...' \
#     bash scripts/monitor/watch_snt_tnt_acc70_train_eval_full8.sh
set -euo pipefail

POLL_SEC=${POLL_SEC:-180}
STEP=${STEP:-100}
BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

if [[ -n "${SPECS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  SPECS=(${SPECS_OVERRIDE})
else
  SPECS=(
    "3122844|qwen3_1.7b|snt_tnt_1e_6_ot_acc70_1p7b|snt_tnt_acc70_1p7b_ckpt100|snt_tnt_acc70_1p7b|1.7b"
    "3122883|qwen3_4b|snt_tnt_1e_6_ot_acc70_4b|snt_tnt_acc70_4b_ckpt100|snt_tnt_acc70_4b|4b"
  )
fi

EVAL_DATASETS=(aime24 aime25 aime26 hmmt25)
EVAL_MODES=(nothink think)

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

submit_evals() {
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
  for ds in "${EVAL_DATASETS[@]}"; do
    for mode in "${EVAL_MODES[@]}"; do
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

watch_one() {
  local train_jid="$1" model_tag="$2" run_name="$3" eval_tag="$4" prefix="$5" eval_subdir="$6"
  local output_dir="${BASE_DIR}/outputs/${model_tag}/${run_name}/${train_jid}"
  local ckpt="${output_dir}/checkpoint-${STEP}"
  local waited=0

  log "[${train_jid}] watch start model=${model_tag} run=${run_name} ckpt=${ckpt} tag=${eval_tag}"

  while true; do
    if ckpt_ready "${ckpt}"; then
      log "[${train_jid}] checkpoint-${STEP} ready"
      submit_evals "${train_jid}" "${ckpt}" "${eval_tag}" "${prefix}" "${eval_subdir}"
      return 0
    fi

    if ! job_in_queue "${train_jid}"; then
      local st
      st=$(job_state "${train_jid}")
      log "[${train_jid}] left queue; state=${st}"
      sleep 15
      if ckpt_ready "${ckpt}"; then
        log "[${train_jid}] ckpt found after job end; submitting"
        submit_evals "${train_jid}" "${ckpt}" "${eval_tag}" "${prefix}" "${eval_subdir}"
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
  log "snt_tnt_acc70 monitor start: ${#SPECS[@]} jobs × 8 evals, poll=${POLL_SEC}s step=${STEP}"
  local pids=() spec jid model_tag run_name eval_tag prefix eval_subdir rc=0

  for spec in "${SPECS[@]}"; do
    [[ -n "${spec}" ]] || continue
    IFS='|' read -r jid model_tag run_name eval_tag prefix eval_subdir <<<"${spec}"
    watch_one "${jid}" "${model_tag}" "${run_name}" "${eval_tag}" "${prefix}" "${eval_subdir}" &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  log "snt_tnt_acc70 monitor done (rc=${rc})"
  return "${rc}"
}

main "$@"
