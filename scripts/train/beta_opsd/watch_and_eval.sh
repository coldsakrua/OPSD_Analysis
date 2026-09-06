#!/bin/bash
# CPU-node watcher: when each β-OPSD LoRA train has checkpoint-200, submit 4 evals.
# Does NOT use final/.
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
cd "${BASE_DIR}"
INTERVAL=${INTERVAL:-120}
SUBMIT="${BASE_DIR}/scripts/train/beta_opsd/submit_four_evals.sh"
STATE_DIR="${BASE_DIR}/log/train/beta_opsd/watch_state"
LOG="${BASE_DIR}/log/train/beta_opsd/watch_eval.${SLURM_JOB_ID:-manual}.log"
mkdir -p "${STATE_DIR}" "$(dirname "${LOG}")"
chmod +x "${SUBMIT}" \
  "${BASE_DIR}/scripts/train/beta_opsd/merge_olmo_lora.sh" 2>/dev/null || true

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

# MODEL_KEY | MODEL_TAG | RUN_NAME
declare -a RUNS=(
  "qwen3_1.7b|qwen3_1.7b|beta_opsd_w05to08_rtg099_lora_lr5e6_ot_qwen3_1.7b"
  "qwen3_4b_thinking|qwen3_4b_thinking|beta_opsd_w05to08_rtg099_lora_lr5e6_ot_qwen3_4b_thinking"
  "qwen3_4b_instruct|qwen3_4b_instruct|beta_opsd_w05to08_rtg099_lora_lr5e6_ot_qwen3_4b_instruct"
  "olmo3_7b_think|olmo3_7b_think|beta_opsd_w05to08_rtg099_lora_lr5e6_ot_olmo3_7b_think"
  "olmo3_7b_instruct|olmo3_7b_instruct|beta_opsd_w05to08_rtg099_lora_lr5e6_ot_olmo3_7b_instruct"
)

find_ckpt200() {
  local model_tag="$1" run_name="$2"
  local root="${BASE_DIR}/outputs/${model_tag}/${run_name}"
  [[ -d "${root}" ]] || return 1
  local d
  for d in $(ls -1d "${root}"/*/ 2>/dev/null | sort -V -r); do
    if [[ -f "${d}checkpoint-200/adapter_config.json" ]] && \
       { [[ -f "${d}checkpoint-200/adapter_model.safetensors" ]] || [[ -f "${d}checkpoint-200/adapter_model.bin" ]]; }; then
      echo "${d}checkpoint-200"
      return 0
    fi
  done
  return 1
}

log "host=$(hostname) job=${SLURM_JOB_ID:-none} interval=${INTERVAL}s"
log "watching 5 β-OPSD runs → eval on checkpoint-200 only (no final)"

done_count=0
while true; do
  done_count=0
  pending=0
  for entry in "${RUNS[@]}"; do
    IFS='|' read -r model_key model_tag run_name <<<"${entry}"
    stamp="${STATE_DIR}/${model_key}.done"
    if [[ -f "${stamp}" ]]; then
      done_count=$((done_count + 1))
      continue
    fi
    if ckpt=$(find_ckpt200 "${model_tag}" "${run_name}"); then
      log "READY ${model_key} -> ${ckpt}"
      if MODEL_KEY="${model_key}" CHECKPOINT_PATH="${ckpt}" \
          EVAL_TAG="beta_opsd_lora_${model_tag}_ckpt200" \
          bash "${SUBMIT}" >>"${LOG}" 2>&1; then
        date -Is >"${stamp}"
        done_count=$((done_count + 1))
        log "SUBMITTED evals for ${model_key}"
      else
        log "ERROR submitting ${model_key} (will retry)"
        pending=$((pending + 1))
      fi
    else
      pending=$((pending + 1))
      log "WAIT ${model_key} (no checkpoint-200 yet)"
    fi
  done

  log "progress ${done_count}/5 ready+submitted; pending=${pending}"
  if [[ "${done_count}" -ge 5 ]]; then
    log "ALL 5 runs have evals submitted; exiting"
    break
  fi
  sleep "${INTERVAL}"
done

log "watch finished"
