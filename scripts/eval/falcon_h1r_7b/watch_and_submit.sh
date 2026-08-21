#!/bin/bash
# Watch the Falcon-H1R-7B download directory. When the total size is unchanged
# for STABLE_SECS and the required weight shards are in place, submit the four
# eval jobs via submit_four.sh.
#
# Usage (from OPSD_Analysis, or anywhere):
#   bash scripts/eval/falcon_h1r_7b/watch_and_submit.sh
#   INTERVAL=60 STABLE_SECS=180 bash scripts/eval/falcon_h1r_7b/watch_and_submit.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/falcon-h1r-7b}
INTERVAL=${INTERVAL:-30}
STABLE_SECS=${STABLE_SECS:-120}
# 4 shards on ModelScope are ~4.65G + 4.61G + 4.62G + 260M ≈ 14G.
MIN_BYTES=${MIN_BYTES:-12000000000}
LOG_DIR="${BASE_DIR}/log/eval/falcon_h1r_7b"
LOG_FILE="${LOG_DIR}/watch_and_submit.log"
STAMP_FILE="${LOG_DIR}/watch_and_submit.submitted"
PID_FILE="${LOG_DIR}/watch_and_submit.pid"

REQUIRED_FILES=(
  config.json
  tokenizer.json
  model.safetensors.index.json
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
)

mkdir -p "${LOG_DIR}"
cd "${BASE_DIR}"
echo $$ > "${PID_FILE}"

human() {
  awk -v b="${1:-0}" 'BEGIN {
    if (b < 1024) printf "%dB", b
    else if (b < 1048576) printf "%.1fKiB", b / 1024
    else if (b < 1073741824) printf "%.2fMiB", b / 1048576
    else printf "%.2fGiB", b / 1073741824
  }'
}

dir_bytes() {
  du -sb "$1" 2>/dev/null | awk '{print $1}'
}

log() {
  echo "[watch $(date '+%F %T')] $*" | tee -a "${LOG_FILE}"
}

checkpoint_ready() {
  local f tsize
  for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -s "${CHECKPOINT_PATH}/${f}" ]]; then
      return 1
    fi
  done
  # ModelScope keeps in-progress shards under ._____temp; treat leftover
  # payload there as "still downloading".
  if [[ -d "${CHECKPOINT_PATH}/._____temp" ]]; then
    tsize=$(dir_bytes "${CHECKPOINT_PATH}/._____temp")
    tsize=${tsize:-0}
    if (( tsize > 1048576 )); then
      return 1
    fi
  fi
  return 0
}

if [[ -f "${STAMP_FILE}" && "${FORCE:-0}" != "1" ]]; then
  log "already submitted ($(cat "${STAMP_FILE}")); set FORCE=1 to submit again"
  rm -f "${PID_FILE}"
  exit 0
fi

log "watching ${CHECKPOINT_PATH}"
log "interval=${INTERVAL}s stable_secs=${STABLE_SECS} min_bytes=$(human "${MIN_BYTES}")"
log "log_file=${LOG_FILE}"

last_size=""
last_change_ts=$(date +%s)

while true; do
  if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
    log "checkpoint dir missing; wait"
    sleep "${INTERVAL}"
    continue
  fi

  size=$(dir_bytes "${CHECKPOINT_PATH}")
  size=${size:-0}
  now=$(date +%s)

  if [[ "${size}" != "${last_size}" ]]; then
    delta=""
    if [[ -n "${last_size}" ]]; then
      delta=$((size - last_size))
      log "size changed $(human "${last_size}") -> $(human "${size}") (delta=$(human "${delta}"))"
    else
      log "size $(human "${size}")"
    fi
    last_size=${size}
    last_change_ts=${now}
  fi

  stable=$((now - last_change_ts))
  ready=0
  if checkpoint_ready; then
    ready=1
  fi

  missing=()
  for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -s "${CHECKPOINT_PATH}/${f}" ]]; then
      missing+=("${f}")
    fi
  done
  miss_msg="complete"
  if ((${#missing[@]} > 0)); then
    miss_msg="missing ${missing[*]}"
  fi

  if (( ready == 1 && size >= MIN_BYTES && stable >= STABLE_SECS )); then
    log "download complete: size=$(human "${size}") unchanged for ${stable}s; ${miss_msg}"
    break
  fi

  log "wait size=$(human "${size}") stable=${stable}s/${STABLE_SECS}s ready=${ready} ${miss_msg}"
  sleep "${INTERVAL}"
done

log "submitting four eval jobs"
set +e
submit_out=$("${SCRIPT_DIR}/submit_four.sh" 2>&1)
submit_rc=$?
set -e
log "${submit_out}"
if (( submit_rc != 0 )); then
  log "error: submit_four.sh exited ${submit_rc}"
  rm -f "${PID_FILE}"
  exit "${submit_rc}"
fi

{
  echo "time=$(date '+%F %T')"
  echo "${submit_out}"
} > "${STAMP_FILE}"
log "submitted"
rm -f "${PID_FILE}"
