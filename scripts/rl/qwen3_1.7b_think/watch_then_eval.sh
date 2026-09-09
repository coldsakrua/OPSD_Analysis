#!/bin/bash
# CPU-node watcher: when GRPO train finishes, merge FSDP→HF then submit 4 think evals
# (aime24/25/26 + hmmt25).
#
# Manifest TSV columns:
#   model_key  model_tag  run_name  train_jid  target_step  seed  eval_tag
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
cd "${BASE_DIR}"

MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/rl/qwen3_1.7b_think/watch_grpo_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}
MERGE_SCRIPT="${BASE_DIR}/scripts/eval/qwen3_4b_base/merge_fsdp_ckpt.sh"
SUBMIT="${BASE_DIR}/scripts/eval/1.7b/seed/submit_four_seed.sh"
STATE_DIR="${BASE_DIR}/log/train/rl/qwen3_1.7b_think/watch_state"
LOG="${BASE_DIR}/log/train/rl/qwen3_1.7b_think/watch.${SLURM_JOB_ID:-manual}.log"
mkdir -p "${STATE_DIR}" "$(dirname "${LOG}")"
chmod +x "${SUBMIT}" "${MERGE_SCRIPT}" 2>/dev/null || true

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

actor_dir_ready() {
  local actor=$1
  [[ -d "${actor}" ]] || return 1
  [[ -f "${actor}/fsdp_config.json" ]] || return 1
  compgen -G "${actor}/model_world_size_*_rank_*.pt" >/dev/null
}

hf_merged_ready() {
  local p=$1
  [[ -f "${p}/config.json" ]] || return 1
  [[ -f "${p}/model.safetensors" ]] \
    || [[ -f "${p}/model.safetensors.index.json" ]] \
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

find_actor() {
  local model_tag=$1 run_name=$2 train_jid=$3 target_step=$4
  local root="${BASE_DIR}/outputs/${model_tag}/${run_name}/${train_jid}"
  local preferred="${root}/global_step_${target_step}/actor"
  if actor_dir_ready "${preferred}"; then
    echo "${preferred}"
    return 0
  fi
  # Fallback: highest global_step_* with a ready actor
  local best="" best_n=-1
  local d step n
  for d in "${root}"/global_step_*/actor; do
    [[ -d "${d}" ]] || continue
    actor_dir_ready "${d}" || continue
    step="$(basename "$(dirname "${d}")")"
    n="${step#global_step_}"
    [[ "${n}" =~ ^[0-9]+$ ]] || continue
    if (( n > best_n )); then
      best_n=$n
      best=$d
    fi
  done
  [[ -n "${best}" ]] || return 1
  echo "${best}"
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

log "host=$(hostname) job=${SLURM_JOB_ID:-none} interval=${INTERVAL}s post_wait=${POST_DONE_WAIT}s"
log "manifest=${MANIFEST} rows=${total}"

while true; do
  done_count=0
  pending=0
  failed=0

  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r model_key model_tag run_name train_jid target_step seed eval_tag <<<"${row}"
    stamp="${STATE_DIR}/${model_key}_${train_jid}.submitted"
    fail_stamp="${STATE_DIR}/${model_key}_${train_jid}.failed"

    if [[ -f "${stamp}" ]]; then
      done_count=$((done_count + 1))
      continue
    fi
    if [[ -f "${fail_stamp}" ]] && ! find_actor "${model_tag}" "${run_name}" "${train_jid}" "${target_step}" >/dev/null; then
      failed=$((failed + 1))
      continue
    fi

    if job_in_queue "${train_jid}"; then
      st=$(job_state "${train_jid}")
      pending=$((pending + 1))
      log "WAIT ${model_key} train=${train_jid} state=${st}"
      continue
    fi

    st=$(job_state "${train_jid}")

    if ! actor=$(find_actor "${model_tag}" "${run_name}" "${train_jid}" "${target_step}"); then
      case "${st}" in
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
          log "FAIL ${model_key} train=${train_jid} state=${st} no usable actor ckpt"
          date -Is >"${fail_stamp}"
          echo "${st}" >>"${fail_stamp}"
          failed=$((failed + 1))
          ;;
        *)
          pending=$((pending + 1))
          log "WAIT ${model_key} train=${train_jid} left queue (state=${st}) but actor not ready"
          ;;
      esac
      continue
    fi

    log "READY actor=${actor} (train_state=${st})"
    sleep "${POST_DONE_WAIT}"
    if ! actor=$(find_actor "${model_tag}" "${run_name}" "${train_jid}" "${target_step}"); then
      pending=$((pending + 1))
      log "WAIT ${model_key} actor disappeared after wait"
      continue
    fi

    merged="${actor}/hf_merged"
    if ! hf_merged_ready "${merged}"; then
      log "MERGE start local_dir=${actor} -> ${merged}"
      merge_jid=$(sbatch --parsable --chdir="${BASE_DIR}" --wait \
        --job-name="merge_grpo1p7_${train_jid}" \
        --output="${BASE_DIR}/log/train/rl/qwen3_1.7b_think/merge_%j.out" \
        --export=ALL,BASE_DIR="${BASE_DIR}",LOCAL_DIR="${actor}",TARGET_DIR="${merged}" \
        "${MERGE_SCRIPT}" 2>>"${LOG}" | tail -1 || true)
      merge_jid="${merge_jid%%;*}"
      log "MERGE job=${merge_jid:-?} finished"
      if ! hf_merged_ready "${merged}"; then
        pending=$((pending + 1))
        log "ERROR merge incomplete at ${merged} (will retry)"
        continue
      fi
    else
      log "MERGE skip (already exists): ${merged}"
    fi

    step_name="$(basename "$(dirname "${actor}")")"
    tag="${eval_tag:-grpo_omr_int_1p7b_${step_name}}"
    log "SUBMIT evals ckpt=${merged} seed=${seed} tag=${tag}"
    if CHECKPOINT_PATH="${merged}" EVAL_TAG="${tag}" SEED="${seed}" BASE_DIR="${BASE_DIR}" \
        bash "${SUBMIT}" >>"${LOG}" 2>&1; then
      date -Is >"${stamp}"
      echo "${merged}" >>"${stamp}"
      rm -f "${fail_stamp}"
      done_count=$((done_count + 1))
      log "SUBMITTED evals for ${model_key} tag=${tag}"
    else
      pending=$((pending + 1))
      log "ERROR submitting evals for ${model_key} (will retry)"
    fi
  done

  log "progress done=${done_count}/${total} pending=${pending} failed=${failed}"
  if [[ $((done_count + failed)) -ge "${total}" ]]; then
    log "ALL trains handled (done=${done_count} failed=${failed}); exiting"
    break
  fi
  sleep "${INTERVAL}"
done

log "watch finished"
exit 0
