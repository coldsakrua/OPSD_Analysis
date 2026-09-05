#!/bin/bash
# CPU-node watcher for Purified OPSD LoRA trains → submit 4 evals when ready.
# Tracks 8 runs: 4 models × {answer, solution}.
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
cd "${BASE_DIR}"
INTERVAL=${INTERVAL:-120}
SUBMIT="${BASE_DIR}/scripts/train/purified_pmi/submit_four_evals.sh"
STATE_DIR="${BASE_DIR}/log/train/purified_pmi/watch_state"
LOG="${BASE_DIR}/log/train/purified_pmi/watch_eval.${SLURM_JOB_ID:-manual}.log"
mkdir -p "${STATE_DIR}" "$(dirname "${LOG}")"
chmod +x "${SUBMIT}" \
  "${BASE_DIR}/scripts/train/purified_pmi/merge_olmo_lora.sh" 2>/dev/null || true

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

# MODEL_KEY | PRIV | TRAIN_GLOB under outputs/<model_tag>/
# Prefer newest job dir that has final/ with adapter weights.
declare -a RUNS=(
  "qwen3_1.7b|answer|qwen3_1.7b|pmi_answer_lora_lr5e6_ot_qwen3_1.7b"
  "qwen3_1.7b|solution|qwen3_1.7b|pmi_solution_lora_lr5e6_ot_qwen3_1.7b"
  "qwen3_4b|answer|qwen3_4b|pmi_answer_lora_lr5e6_ot_qwen3_4b"
  "qwen3_4b|solution|qwen3_4b|pmi_solution_lora_lr5e6_ot_qwen3_4b"
  "qwen3_4b_thinking|answer|qwen3_4b_thinking|pmi_answer_lora_lr5e6_ot_qwen3_4b_thinking"
  "qwen3_4b_thinking|solution|qwen3_4b_thinking|pmi_solution_lora_lr5e6_ot_qwen3_4b_thinking"
  "olmo3_7b_think|answer|olmo3_7b_think|pmi_answer_lora_lr5e6_ot_olmo3_7b_think"
  "olmo3_7b_think|solution|olmo3_7b_think|pmi_solution_lora_lr5e6_ot_olmo3_7b_think"
)

find_final() {
  local model_tag="$1" run_name="$2"
  local root="${BASE_DIR}/outputs/${model_tag}/${run_name}"
  [[ -d "${root}" ]] || return 1
  # Prefer dirs with final/adapter; newest job id first.
  local d
  for d in $(ls -1d "${root}"/*/ 2>/dev/null | sort -V -r); do
    if [[ -f "${d}final/adapter_config.json" ]] && \
       { [[ -f "${d}final/adapter_model.safetensors" ]] || [[ -f "${d}final/adapter_model.bin" ]]; }; then
      # Prefer complete 200-step runs: final exists after save_model.
      echo "${d}final"
      return 0
    fi
    if [[ -f "${d}checkpoint-200/adapter_config.json" ]] && \
       { [[ -f "${d}checkpoint-200/adapter_model.safetensors" ]] || [[ -f "${d}checkpoint-200/adapter_model.bin" ]]; }; then
      echo "${d}checkpoint-200"
      return 0
    fi
  done
  return 1
}

train_running() {
  # Any pmi train still in queue?
  squeue -u "${USER}" -h -o '%j %T' 2>/dev/null | grep -E '^pmi_' | grep -vE 'eval|merge|watch' || true
}

log "host=$(hostname) job=${SLURM_JOB_ID:-none} interval=${INTERVAL}s"
log "watching 8 purified-pmi runs → 4 evals each (aime24/25/26/hmmt25)"

done_count=0
while true; do
  done_count=0
  pending=0
  for entry in "${RUNS[@]}"; do
    IFS='|' read -r model_key priv model_tag run_name <<<"${entry}"
    stamp="${STATE_DIR}/${model_key}_${priv}.done"
    if [[ -f "${stamp}" ]]; then
      done_count=$((done_count + 1))
      continue
    fi
    if ckpt=$(find_final "${model_tag}" "${run_name}"); then
      log "READY ${model_key}/${priv} -> ${ckpt}"
      if MODEL_KEY="${model_key}" CHECKPOINT_PATH="${ckpt}" \
          EVAL_TAG="pmi_${priv}_lora_${model_tag}_final" \
          bash "${SUBMIT}" >>"${LOG}" 2>&1; then
        date -Is >"${stamp}"
        done_count=$((done_count + 1))
        log "SUBMITTED evals for ${model_key}/${priv}"
      else
        log "ERROR submitting ${model_key}/${priv} (will retry)"
        pending=$((pending + 1))
      fi
    else
      pending=$((pending + 1))
      log "WAIT ${model_key}/${priv} (no final/ckpt200 yet)"
    fi
  done

  log "progress ${done_count}/8 ready+submitted; pending=${pending}"
  if [[ "${done_count}" -ge 8 ]]; then
    log "ALL 8 runs have evals submitted; exiting"
    break
  fi

  # If nothing left in train queue and still missing, keep waiting a bit
  # (olmo may be Priority-queued).
  sleep "${INTERVAL}"
done

log "watch finished"
