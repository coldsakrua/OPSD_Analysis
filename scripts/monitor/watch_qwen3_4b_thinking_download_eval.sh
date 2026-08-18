#!/bin/bash
# Watch Qwen3-4B-Thinking-2507 download; when complete, submit 4 think evals.
#
# Usage:
#   nohup bash scripts/monitor/watch_qwen3_4b_thinking_download_eval.sh \
#     > log/monitor/watch_qwen3_4b_thinking_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Overrides:
#   POLL_SEC=300 MODEL_DIR=/path/to/model bash scripts/monitor/watch_qwen3_4b_thinking_download_eval.sh
set -euo pipefail

POLL_SEC=${POLL_SEC:-600}
BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MODEL_DIR=${MODEL_DIR:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b-thinking}
EVAL_TAG=${EVAL_TAG:-qwen3-4b-thinking}
LOCK="${BASE_DIR}/log/monitor/eval_submitted_qwen3_4b_thinking.lock"
JOBS_FILE="${BASE_DIR}/log/monitor/eval_jobs_qwen3_4b_thinking.txt"

log() {
  echo "[$(date '+%F %T')] $*"
}

model_ready() {
  local dir="$1"
  local index="${dir}/model.safetensors.index.json"
  local shard missing=0 size_sum=0 expected=""

  [[ -d "${dir}" ]] || return 1
  [[ -f "${dir}/config.json" ]] || return 1
  [[ -f "${dir}/tokenizer.json" || -f "${dir}/tokenizer_config.json" ]] || return 1
  [[ -f "${index}" ]] || return 1

  # Reject in-progress ModelScope temp downloads when present and non-empty.
  if [[ -d "${dir}/._____temp" ]] && find "${dir}/._____temp" -type f 2>/dev/null | grep -q .; then
    return 1
  fi

  while IFS= read -r shard; do
    [[ -n "${shard}" ]] || continue
    if [[ ! -f "${dir}/${shard}" ]]; then
      missing=1
      break
    fi
    # Skip zero-byte / tiny incomplete shards.
    if [[ "$(stat -c%s "${dir}/${shard}" 2>/dev/null || echo 0)" -lt 1048576 ]]; then
      missing=1
      break
    fi
    size_sum=$((size_sum + $(stat -c%s "${dir}/${shard}")))
  done < <(
    python - "${index}" <<'PY'
import json, sys
idx = json.load(open(sys.argv[1]))
for name in sorted(set(idx["weight_map"].values())):
    print(name)
PY
  )

  [[ "${missing}" -eq 0 ]] || return 1

  expected=$(
    python - "${index}" <<'PY'
import json, sys
idx = json.load(open(sys.argv[1]))
print(idx.get("metadata", {}).get("total_size", 0))
PY
  )
  if [[ -n "${expected}" && "${expected}" -gt 0 ]]; then
    # Allow small filesystem accounting slack (1 MiB).
    if [[ "${size_sum}" -lt $((expected - 1048576)) ]]; then
      return 1
    fi
  fi
  return 0
}

download_status() {
  local dir="$1"
  local index="${dir}/model.safetensors.index.json"
  local have=0 need=0

  if [[ ! -f "${index}" ]]; then
    echo "index missing"
    return
  fi
  need=$(
    python - "${index}" <<'PY'
import json, sys
idx = json.load(open(sys.argv[1]))
print(len(set(idx["weight_map"].values())))
PY
  )
  have=$(
    python - "${index}" "${dir}" <<'PY'
import json, sys, os
idx = json.load(open(sys.argv[1]))
root = sys.argv[2]
n = 0
for name in set(idx["weight_map"].values()):
    p = os.path.join(root, name)
    if os.path.isfile(p) and os.path.getsize(p) >= 1048576:
        n += 1
print(n)
PY
  )
  echo "shards ${have}/${need}"
}

submit_evals() {
  local ds script job_name job_id
  mkdir -p "${BASE_DIR}/log/monitor"
  if [[ -f "${LOCK}" ]]; then
    log "evals already submitted; skip (${LOCK})"
    return 0
  fi
  : > "${LOCK}"
  : > "${JOBS_FILE}"

  cd "${BASE_DIR}"
  log "submitting 4× qwen3_4b_thinking evals model=${MODEL_DIR} tag=${EVAL_TAG}"
  for ds in aime24 aime25 aime26 hmmt25; do
    script="${BASE_DIR}/scripts/eval/qwen3_4b_thinking/${ds}_think.sh"
    [[ -f "${script}" ]] || { log "ERROR: missing ${script}"; return 1; }
    case "${ds}" in
      aime24) job_name=eval_4bt_a24_th ;;
      aime25) job_name=eval_4bt_a25_th ;;
      aime26) job_name=eval_4bt_a26_th ;;
      hmmt25) job_name=eval_4bt_h25_th ;;
    esac
    job_id=$(
      sbatch --parsable \
        --job-name="${job_name}" \
        --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${MODEL_DIR}",EVAL_TAG="${EVAL_TAG}" \
        "${script}" 2>/dev/null | tail -1
    )
    log "submitted ${ds}_think: job ${job_id} (${job_name})"
    echo "${job_id} ${job_name} ${EVAL_TAG} scripts/eval/qwen3_4b_thinking/${ds}_think.sh" >> "${JOBS_FILE}"
  done
  log "done. job ids -> ${JOBS_FILE}"
}

mkdir -p "${BASE_DIR}/log/monitor"
log "watch start BASE_DIR=${BASE_DIR}"
log "MODEL_DIR=${MODEL_DIR} POLL_SEC=${POLL_SEC}"

if [[ -f "${LOCK}" ]]; then
  log "lock exists; evals already submitted earlier. exit."
  exit 0
fi

while true; do
  if model_ready "${MODEL_DIR}"; then
    log "model ready ($(download_status "${MODEL_DIR}"))"
    submit_evals
    log "monitor finished"
    exit 0
  fi
  log "not ready yet ($(download_status "${MODEL_DIR}")); sleep ${POLL_SEC}s"
  sleep "${POLL_SEC}"
done
