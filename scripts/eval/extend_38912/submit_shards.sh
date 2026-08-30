#!/bin/bash
# Submit all extend shards (or a subset).
# Usage:
#   bash scripts/eval/extend_38912/submit_shards.sh
#   bash scripts/eval/extend_38912/submit_shards.sh shard_00.jsonl   # smoke one shard
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SHARD_DIR="${BASE_DIR}/scripts/eval/extend_38912/manifest/shards"
WORKER="${SCRIPT_DIR}/extend_truncated_worker.sh"

cd "${BASE_DIR}"
mkdir -p log/eval/extend_38912/worker

if [[ $# -gt 0 ]]; then
  FILES=("$@")
else
  mapfile -t FILES < <(ls -1 "${SHARD_DIR}"/shard_*.jsonl | sort)
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "[submit] no shards found; run build_manifest.py first" >&2
  exit 1
fi

for f in "${FILES[@]}"; do
  if [[ "${f}" != /* ]]; then
    if [[ -f "${SHARD_DIR}/${f}" ]]; then
      f="${SHARD_DIR}/${f}"
    else
      f="${BASE_DIR}/${f}"
    fi
  fi
  if [[ ! -f "${f}" ]]; then
    echo "[submit] missing shard: ${f}" >&2
    exit 1
  fi
  name="$(basename "${f}" .jsonl)"
  jid=$(SHARD_FILE="${f}" sbatch --parsable --job-name="ext_${name}" "${WORKER}")
  echo "[submit] ${name} -> job ${jid}  shard=${f}"
done
