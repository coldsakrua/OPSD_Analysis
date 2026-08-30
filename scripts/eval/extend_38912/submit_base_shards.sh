#!/bin/bash
# Submit base-family extend shards (qwen3_1.7b_base / qwen3_4b_base).
# Usage (from OPSD_Analysis):
#   bash scripts/eval/extend_38912/submit_base_shards.sh
#   bash scripts/eval/extend_38912/submit_base_shards.sh --build   # rebuild manifest then submit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKER="${SCRIPT_DIR}/extend_truncated_worker.sh"
LIST="${SCRIPT_DIR}/manifest_base/submit_list.txt"
JIDS="${SCRIPT_DIR}/manifest_base/submit_jids.txt"

cd "${BASE_DIR}"
mkdir -p log/eval/extend_38912/worker

if [[ "${1:-}" == "--build" ]]; then
  python "${SCRIPT_DIR}/build_manifest_base.py" --base-dir "${BASE_DIR}"
fi

if [[ ! -f "${LIST}" ]]; then
  echo "[submit-base] missing ${LIST}; run with --build first" >&2
  exit 1
fi

: > "${JIDS}"
while IFS=$'\t' read -r shard_path bs name; do
  [[ -z "${shard_path:-}" ]] && continue
  jname="ext_${name}"
  jname="${jname:0:60}"
  jid=$(sbatch --parsable \
    --job-name="${jname}" \
    --export=ALL,SHARD_FILE="${shard_path}",GENERATE_BATCH_SIZE="${bs}",GPU_MEM=0.85,ALLOW_LONG_MAX_MODEL_LEN=1 \
    "${WORKER}")
  echo "[submit-base] ${name} bs=${bs} -> ${jid}"
  echo -e "${jid}\t${bs}\t${name}\t${shard_path}" >> "${JIDS}"
done < "${LIST}"

echo "[submit-base] wrote ${JIDS} (n=$(wc -l < "${JIDS}"))"
