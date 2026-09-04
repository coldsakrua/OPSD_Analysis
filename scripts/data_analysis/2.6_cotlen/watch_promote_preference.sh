#!/bin/bash
# Watch cotlen easy/hard gen jobs: when rollouts (+accuracy) are ready, cancel the
# leftover preference phase on the old coupled job and submit preference separately.
#
# Usage:
#   ./watch_promote_preference.sh              # default watch list
#   ./watch_promote_preference.sh --once       # single pass
#   INTERVAL=60 ./watch_promote_preference.sh
#
# State: log/data_analysis/26/watch_promote.state
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
DIR="${BASE_DIR}/scripts/data_analysis/2.6_cotlen"
LOG_DIR="${BASE_DIR}/log/data_analysis/26"
STATE_FILE="${LOG_DIR}/watch_promote.state"
SUBMIT_PREF="${DIR}/submit_preference.sh"
INTERVAL=${INTERVAL:-60}
ONCE=0
[[ "${1:-}" == "--once" ]] && ONCE=1

mkdir -p "${LOG_DIR}"
touch "${STATE_FILE}"

log() { echo "[$(date '+%F %T')] $*"; }

# job_id|output_dir  (pipe-separated; avoid colon in paths)
# Default: all currently known cotlen gen jobs that still couple preference.
DEFAULT_WATCH=(
  "3421150|${BASE_DIR}/scripts/data_analysis/outputs/cotlen/qwen3_4b_instruct/snt_tnt_hard_3421150"
  "3421170|${BASE_DIR}/scripts/data_analysis/outputs/cotlen/qwen3_1.7b/st_tt_easy_3421170"
  "3421171|${BASE_DIR}/scripts/data_analysis/outputs/cotlen/qwen3_1.7b/st_tt_hard_3421171"
  "3421172|${BASE_DIR}/scripts/data_analysis/outputs/cotlen/qwen3_4b/st_tt_easy_3421172"
  "3421173|${BASE_DIR}/scripts/data_analysis/outputs/cotlen/qwen3_4b/st_tt_hard_3421173"
  "3423005|${BASE_DIR}/scripts/data_analysis/outputs/cotlen/olmo3_7b_think/st_tt_easy_3423005"
  "3423006|${BASE_DIR}/scripts/data_analysis/outputs/cotlen/olmo3_7b_think/st_tt_hard_3423006"
  "3423007|${BASE_DIR}/scripts/data_analysis/outputs/cotlen/olmo3_7b_instruct/snt_tnt_easy_3423007"
  "3423008|${BASE_DIR}/scripts/data_analysis/outputs/cotlen/olmo3_7b_instruct/snt_tnt_hard_3423008"
)

# Also promote dirs that already finished gen (no active job) if listed via EXTRA_DIRS
EXTRA_DIRS=(
  "${BASE_DIR}/scripts/data_analysis/outputs/cotlen/qwen3_4b_instruct/snt_tnt_easy_3421149"
)

is_promoted() {
  local key="$1"
  grep -qxF "${key}" "${STATE_FILE}" 2>/dev/null
}

mark_promoted() {
  local key="$1"
  echo "${key}" >> "${STATE_FILE}"
}

job_state() {
  local jid="$1"
  squeue -j "${jid}" -h -o '%T' 2>/dev/null || echo "GONE"
}

rollouts_ready() {
  local out="$1"
  [[ -f "${out}/rollouts.jsonl" ]] || return 1
  local n
  n=$(wc -l < "${out}/rollouts.jsonl")
  [[ "${n}" -gt 0 ]]
}

has_preference() {
  local out="$1"
  [[ -f "${out}/summary_preference.json" ]] && return 0
  [[ -f "${out}/rollout_metrics.jsonl" ]] || return 1
  local n
  n=$(wc -l < "${out}/rollout_metrics.jsonl")
  [[ "${n}" -gt 100 ]]  # partial score from cancelled job — treat as incomplete
}

in_score_phase() {
  local jid="$1"
  local f
  f=$(ls "${LOG_DIR}"/*."${jid}".out 2>/dev/null | head -1 || true)
  [[ -n "${f}" ]] || return 1
  tr '\r' '\n' < "${f}" | rg -q 'score/teacher_prompt:|===== phase 2: score'
}

ensure_accuracy() {
  local out="$1"
  [[ -f "${out}/accuracy_summary.json" ]] && return 0
  # Best-effort: run accuracy-only on login node if rollouts exist (CPU, fast).
  log "accuracy missing under ${out}; will rely on preference job RUN_ACCURACY or existing summary"
  return 0
}

promote() {
  local jid="$1"
  local out="$2"
  local key="pref:${out}"

  if is_promoted "${key}"; then
    log "skip already promoted: ${out}"
    return 0
  fi
  if has_preference "${out}" && [[ ! -f "${out}/.preference_incomplete" ]]; then
    # If metrics look complete vs rollouts, skip
    local nr nm
    nr=$(wc -l < "${out}/rollouts.jsonl")
    nm=0
    [[ -f "${out}/rollout_metrics.jsonl" ]] && nm=$(wc -l < "${out}/rollout_metrics.jsonl")
    if [[ "${nm}" -ge "${nr}" ]]; then
      log "skip preference already complete (${nm}/${nr}): ${out}"
      mark_promoted "${key}"
      return 0
    fi
  fi

  if ! rollouts_ready "${out}"; then
    log "not ready (no rollouts): job=${jid} out=${out}"
    return 1
  fi

  ensure_accuracy "${out}"

  local st
  st=$(job_state "${jid}")
  if [[ "${st}" == "RUNNING" || "${st}" == "PENDING" || "${st}" == "COMPLETING" ]]; then
    log "scancel ${jid} (state=${st}) — gen done, hand off preference"
    scancel "${jid}" || true
    sleep 2
  else
    log "job ${jid} already ${st}; submitting preference only"
  fi

  # Keep partial rollout_metrics*.jsonl for resume on preference-only reruns.
  rm -f "${out}/summary_preference.json" 2>/dev/null || true

  log "submit preference for ${out}"
  if ! bash "${SUBMIT_PREF}" "${out}"; then
    log "ERROR: submit_preference failed for ${out}"
    return 1
  fi
  mark_promoted "${key}"
  log "promoted OK: ${out}"
}

should_promote_job() {
  local jid="$1"
  local out="$2"
  rollouts_ready "${out}" || return 1
  local st
  st=$(job_state "${jid}")
  # Gen finished if rollouts exist. Promote when scoring started, or job ended without full preference.
  if in_score_phase "${jid}"; then
    return 0
  fi
  if [[ "${st}" == "GONE" || "${st}" == "COMPLETED" || "${st}" == "FAILED" || "${st}" == "CANCELLED" || "${st}" == "TIMEOUT" ]]; then
    has_preference "${out}" && return 1
    return 0
  fi
  # Still generating: rollouts file usually appears only at end of phase1 — if present while RUNNING,
  # phase1 is done and phase2 may be starting or about to start.
  if [[ "${st}" == "RUNNING" ]] && rollouts_ready "${out}"; then
    # Wait until accuracy exists or score phase visible (phase1 fully flushed)
    [[ -f "${out}/accuracy_summary.json" ]] && return 0
    in_score_phase "${jid}" && return 0
  fi
  return 1
}

pass_once() {
  local entry jid out
  for entry in "${DEFAULT_WATCH[@]}"; do
    jid="${entry%%|*}"
    out="${entry#*|}"
    if should_promote_job "${jid}" "${out}"; then
      promote "${jid}" "${out}" || true
    else
      st=$(job_state "${jid}")
      n=0
      [[ -f "${out}/rollouts.jsonl" ]] && n=$(wc -l < "${out}/rollouts.jsonl")
      log "watch job=${jid} state=${st} rollouts=${n} out=$(basename "${out}")"
    fi
  done
  for out in "${EXTRA_DIRS[@]}"; do
    key="pref:${out}"
    is_promoted "${key}" && continue
    rollouts_ready "${out}" || continue
    # easy already has pending 3428012 preference — mark if metrics incomplete but job pending
    if squeue -u "${USER}" -h -o '%j %T' 2>/dev/null | rg -q 'da26_snt_tnt_4bi_easy.*(PENDING|RUNNING)'; then
      log "easy preference job already queued; mark ${out}"
      mark_promoted "${key}"
      continue
    fi
    has_preference "${out}" && continue
    promote "none" "${out}" || true
  done
}

log "start watch INTERVAL=${INTERVAL}s state=${STATE_FILE}"
while true; do
  pass_once
  [[ "${ONCE}" == "1" ]] && break
  # stop when nothing left to watch
  remaining=0
  for entry in "${DEFAULT_WATCH[@]}"; do
    jid="${entry%%|*}"
    out="${entry#*|}"
    key="pref:${out}"
    is_promoted "${key}" && continue
    st=$(job_state "${jid}")
    if [[ "${st}" != "GONE" ]] || { rollouts_ready "${out}" && ! has_preference "${out}"; }; then
      remaining=$((remaining + 1))
    fi
  done
  if [[ "${remaining}" -eq 0 ]]; then
    log "all watched jobs promoted or idle; exiting"
    break
  fi
  sleep "${INTERVAL}"
done
log "watch done"
