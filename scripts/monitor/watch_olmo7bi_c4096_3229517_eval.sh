#!/bin/bash
# Watch olmo3-7B-Instruct snt_tnt clip005 c4096 train (default: 3229517).
# When checkpoint-100 is ready, submit 4 SGLang evals: aime24/25/26 + hmmt25.
#
# Usage:
#   nohup bash scripts/monitor/watch_olmo7bi_c4096_3229517_eval.sh \
#     > log/monitor/watch_olmo7bi_c4096_3229517_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Overrides:
#   TRAIN_JID=3229517 STEP=100 POLL_SEC=120 \
#     bash scripts/monitor/watch_olmo7bi_c4096_3229517_eval.sh
set -euo pipefail

POLL_SEC=${POLL_SEC:-120}
STEP=${STEP:-100}
TRAIN_JID=${TRAIN_JID:-3229517}
MODEL_TAG=${MODEL_TAG:-olmo3_7b_instruct}
RUN_NAME=${RUN_NAME:-snt_tnt_clip005_1e_6_oti_olmo7bi_c4096}
EVAL_TAG=${EVAL_TAG:-snt_tnt_clip005_olmo7bi_c4096_ckpt100}
PREFIX=${PREFIX:-snt_tnt_clip005_c4096_olmo7bi}
BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CKPT=${CKPT:-${BASE_DIR}/outputs/${MODEL_TAG}/${RUN_NAME}/${TRAIN_JID}/checkpoint-${STEP}}

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

submit_olmo_sgl4() {
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
  log "[${train_jid}] submitting 4 olmo3_7b_instruct sgl evals ckpt=${ckpt} tag=${eval_tag}"
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
      local final_ckpt
      final_ckpt="$(dirname "${ckpt}")/final"
      if ckpt_ready "${final_ckpt}"; then
        log "[${train_jid}] using final/ instead of checkpoint-${STEP}"
        CKPT="${final_ckpt}"
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

main() {
  mkdir -p "${BASE_DIR}/log/monitor"
  log "BASE_DIR=${BASE_DIR}"
  log "TRAIN_JID=${TRAIN_JID} RUN_NAME=${RUN_NAME} EVAL_TAG=${EVAL_TAG} PREFIX=${PREFIX}"
  log "expect ckpt=${CKPT}"
  watch_until_ckpt "${TRAIN_JID}" "${CKPT}" || exit 1
  submit_olmo_sgl4 "${TRAIN_JID}" "${CKPT}" "${EVAL_TAG}" "${PREFIX}"
  log "monitor finished ok"
}

main "$@"
