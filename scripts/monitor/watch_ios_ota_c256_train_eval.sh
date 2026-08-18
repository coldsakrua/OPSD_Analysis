#!/bin/bash
# Watch the four ios/ota c256 train jobs; when checkpoint-100 is ready, submit evals.
#
#   3242758  1.7b think iOS  → 4× think (aime24/25/26 + hmmt25)
#   3242759  1.7b think OTA  → 4× think
#   3242815  olmo3-7b-it iOS → 4× sgl
#   3242816  olmo3-7b-it OTA → 4× sgl
#
# Usage:
#   nohup bash scripts/monitor/watch_ios_ota_c256_train_eval.sh \
#     > log/monitor/watch_ios_ota_c256_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Overrides:
#   POLL_SEC=60 STEP=100 bash scripts/monitor/watch_ios_ota_c256_train_eval.sh
set -euo pipefail

POLL_SEC=${POLL_SEC:-120}
STEP=${STEP:-100}
BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

# Spec: JOB_ID|MODEL_TAG|RUN_NAME|EVAL_TAG|JOB_NAME_PREFIX|EVAL_KIND
# EVAL_KIND: think17 | sgl_olmo
# ckpt: outputs/${MODEL_TAG}/${RUN_NAME}/${JOB_ID}/checkpoint-${STEP}
if [[ -n "${SPECS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  SPECS=(${SPECS_OVERRIDE})
else
  SPECS=(
    "3242758|qwen3_1.7b|st_tt_clip005_1e_6_openthoughts_irr_other_sol_1p7b_c256|st_tt_clip005_1e6_ios_1p7b_c256_ckpt100|st_tt_clip005_1e6_ios_c256_1p7b|think17"
    "3242759|qwen3_1.7b|st_tt_clip005_1e_6_openthoughts_answer_1p7b_c256|st_tt_clip005_1e6_ota_1p7b_c256_ckpt100|st_tt_clip005_1e6_ota_c256_1p7b|think17"
    "3242815|olmo3_7b_instruct|snt_tnt_clip005_1e_6_openthoughts_irr_other_sol_olmo7bit_c256|snt_tnt_clip005_1e6_ios_olmo_c256_ckpt100|snt_tnt_clip005_1e6_ios_c256_olmo|sgl_olmo"
    "3242816|olmo3_7b_instruct|snt_tnt_clip005_1e_6_openthoughts_answer_olmo7bit_c256|snt_tnt_clip005_1e6_ota_olmo_c256_ckpt100|snt_tnt_clip005_1e6_ota_c256_olmo|sgl_olmo"
  )
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

submit_think17() {
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
  log "[${train_jid}] submitting 4×1.7b think evals ckpt=${ckpt} tag=${eval_tag}"
  for ds in aime24 aime25 aime26 hmmt25; do
    script="${BASE_DIR}/scripts/eval/1.7b/${ds}_think.sh"
    [[ -f "${script}" ]] || { log "ERROR: missing ${script}"; return 1; }
    job_name="${prefix}_$(short_ds "${ds}")_th"
    job_id=$(
      sbatch --parsable \
        --job-name="${job_name}" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${ckpt}",EVAL_TAG="${eval_tag}" \
        "${script}" 2>/dev/null | tail -1
    )
    log "[${train_jid}] submitted ${ds}_think: job ${job_id} (${job_name})"
    echo "${job_id} ${job_name} ${eval_tag} scripts/eval/1.7b/${ds}_think.sh" >> "${jobs_file}"
  done
  log "[${train_jid}] done. job ids -> ${jobs_file}"
}

submit_sgl_olmo() {
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
  log "[${train_jid}] submitting 4×olmo3_7b_instruct sgl evals ckpt=${ckpt} tag=${eval_tag}"
  for ds in aime24 aime25 aime26 hmmt25; do
    script="${BASE_DIR}/scripts/eval/olmo3_7b_instruct/${ds}_sgl.sh"
    [[ -f "${script}" ]] || { log "ERROR: missing ${script}"; return 1; }
    job_name="${prefix}_$(short_ds "${ds}")_sgl"
    job_id=$(
      sbatch --parsable \
        --job-name="${job_name}" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${ckpt}",EVAL_TAG="${eval_tag}" \
        "${script}" 2>/dev/null | tail -1
    )
    log "[${train_jid}] submitted ${ds}_sgl: job ${job_id} (${job_name})"
    echo "${job_id} ${job_name} ${eval_tag} scripts/eval/olmo3_7b_instruct/${ds}_sgl.sh" >> "${jobs_file}"
  done
  log "[${train_jid}] done. job ids -> ${jobs_file}"
}

submit_for_kind() {
  local kind="$1" train_jid="$2" ckpt="$3" eval_tag="$4" prefix="$5"
  case "${kind}" in
    think17) submit_think17 "${train_jid}" "${ckpt}" "${eval_tag}" "${prefix}" ;;
    sgl_olmo) submit_sgl_olmo "${train_jid}" "${ckpt}" "${eval_tag}" "${prefix}" ;;
    *) log "ERROR: unknown EVAL_KIND=${kind}"; return 1 ;;
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
  log "ios/ota c256 monitor start: ${#SPECS[@]} trains, poll=${POLL_SEC}s step=${STEP}"
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
  log "ios/ota c256 monitor done (rc=${rc})"
  return "${rc}"
}

main "$@"
