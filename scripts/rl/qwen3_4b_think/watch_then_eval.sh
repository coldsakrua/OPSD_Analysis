#!/bin/bash
# CPU-node watcher: wait until global_step_{target_step} is ready (even if train
# is still running), merge FSDP→HF, then submit 4 think evals
# (aime24/25/26 + hmmt25).
#
# Manifest TSV columns:
#   model_key  model_tag  run_name  train_jid  target_step  seed  eval_tag
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
cd "${BASE_DIR}"

MANIFEST=${MANIFEST:-${BASE_DIR}/log/train/rl/qwen3_4b_think/watch_grpo_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}
MERGE_SCRIPT="${BASE_DIR}/scripts/eval/qwen3_4b_base/merge_fsdp_ckpt.sh"
SUBMIT="${BASE_DIR}/scripts/eval/4b/submit_four.sh"
STATE_DIR="${BASE_DIR}/log/train/rl/qwen3_4b_think/watch_state"
LOG="${BASE_DIR}/log/train/rl/qwen3_4b_think/watch.${SLURM_JOB_ID:-manual}.log"
mkdir -p "${STATE_DIR}" "$(dirname "${LOG}")"
chmod +x "${SUBMIT}" "${MERGE_SCRIPT}" 2>/dev/null || true

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

job_state() {
  local jid=$1
  local st
  st=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
  if [[ -n "${st}" ]]; then
    echo "${st}"
    return 0
  fi
  st=$(sacct -j "${jid}" -n -X -o State -P 2>/dev/null | head -1 | tr -d ' ' || true)
  echo "${st:-UNKNOWN}"
}

normalize_state() {
  local st=$1
  st="${st%%+*}"
  echo "${st}"
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

find_target_actor() {
  local model_tag=$1 run_name=$2 train_jid=$3 target_step=$4
  local preferred="${BASE_DIR}/outputs/${model_tag}/${run_name}/${train_jid}/global_step_${target_step}/actor"
  if actor_dir_ready "${preferred}"; then
    echo "${preferred}"
    return 0
  fi
  return 1
}

preserve_conflicting_eval_dir() {
  local tag=$1 merged=$2
  local dir="${BASE_DIR}/eval_outputs/${tag}"
  [[ -d "${dir}" ]] || return 0
  local f
  f=$(compgen -G "${dir}/*.metrics.json" | head -1 || true)
  [[ -n "${f}" ]] || return 0
  if grep -Fq "${merged}" "${f}" 2>/dev/null; then
    return 0
  fi
  local aside="${dir}_prev_$(date +%Y%m%d_%H%M%S)"
  log "RENAME conflicting eval dir ${dir} -> ${aside}"
  mv "${dir}" "${aside}"
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
    stamp="${STATE_DIR}/${model_key}_${train_jid}_gs${target_step}.submitted"
    fail_stamp="${STATE_DIR}/${model_key}_${train_jid}_gs${target_step}.failed"
    st=$(job_state "${train_jid}")
    st_norm=$(normalize_state "${st}")

    if [[ -f "${stamp}" ]]; then
      done_count=$((done_count + 1))
      continue
    fi
    if [[ -f "${fail_stamp}" ]] && ! find_target_actor "${model_tag}" "${run_name}" "${train_jid}" "${target_step}" >/dev/null; then
      failed=$((failed + 1))
      continue
    fi

    if ! actor=$(find_target_actor "${model_tag}" "${run_name}" "${train_jid}" "${target_step}"); then
      case "${st_norm}" in
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
          log "FAIL ${model_key} train=${train_jid} state=${st} no global_step_${target_step}"
          date -Is >"${fail_stamp}"
          echo "${st}" >>"${fail_stamp}"
          failed=$((failed + 1))
          ;;
        *)
          pending=$((pending + 1))
          log "WAIT ${model_key} train=${train_jid} state=${st} for global_step_${target_step}"
          ;;
      esac
      continue
    fi

    log "READY actor=${actor} (train_state=${st} target=gs${target_step})"
    sleep "${POST_DONE_WAIT}"
    if ! actor=$(find_target_actor "${model_tag}" "${run_name}" "${train_jid}" "${target_step}"); then
      pending=$((pending + 1))
      log "WAIT ${model_key} global_step_${target_step} disappeared after wait"
      continue
    fi

    merged="${actor}/hf_merged"
    if ! hf_merged_ready "${merged}"; then
      log "MERGE start local_dir=${actor} -> ${merged}"
      merge_jid=$(sbatch --parsable --chdir="${BASE_DIR}" --wait \
        --job-name="merge_grpo4b_${train_jid}" \
        --output="${BASE_DIR}/log/train/rl/qwen3_4b_think/merge_%j.out" \
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
    tag="${eval_tag:-grpo_omr_int_4b_${step_name}}"
    preserve_conflicting_eval_dir "${tag}" "${merged}"
    log "SUBMIT evals ckpt=${merged} seed=${seed} tag=${tag}"
    if CHECKPOINT_PATH="${merged}" EVAL_TAG="${tag}" BASE_DIR="${BASE_DIR}" \
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
