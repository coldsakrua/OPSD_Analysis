#!/bin/bash
#SBATCH --job-name=lcb_falcon7b_idx_merge
#SBATCH --output=log/eval_code/falcon_h1r_7b/lcb_v6/sgl/%x.%j.out
#SBATCH --partition=C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#
# Merge base + index-slice shard generations, then submit CPU judge.
# Required env: BASE_DIR OUTPUT_JSON DATA_JSONL
# Ranges via SHARD_LIST_FILE (preferred) or SHARD_RANGES.
# Do NOT put comma-separated values in sbatch --export (Slurm splits on ',').
# SHARD_RANGES example: "80:100;100:120;120:140"  (semicolon-separated)
# SHARD_LIST_FILE: one start:end per line
# Optional: BASE_JSONL VAL_N=4 AUTO_JUDGE=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

: "${OUTPUT_JSON:?Set OUTPUT_JSON to final merged metrics path}"
DATA_JSONL=${DATA_JSONL:-${BASE_DIR}/data/livecodebench/test6.jsonl}
BASE_JSONL=${BASE_JSONL:-}
VAL_N=${VAL_N:-4}
SHARD_RANGES=${SHARD_RANGES:-}
SHARD_LIST_FILE=${SHARD_LIST_FILE:-}

cd "${BASE_DIR}"
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval_code/falcon_h1r_7b/lcb_v6/sgl"

ranges=()
if [[ -n "${SHARD_LIST_FILE}" ]]; then
  if [[ ! -f "${SHARD_LIST_FILE}" ]]; then
    echo "[merge] ERROR: missing SHARD_LIST_FILE=${SHARD_LIST_FILE}" >&2
    exit 1
  fi
  echo "[merge] reading ranges from ${SHARD_LIST_FILE}"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    line="${line// /}"
    [[ -z "${line}" ]] && continue
    ranges+=("${line}")
  done < "${SHARD_LIST_FILE}"
elif [[ -n "${SHARD_RANGES}" ]]; then
  # Prefer ';'. Also accept ',' for interactive local runs.
  _sr="${SHARD_RANGES//,/;}"
  IFS=';' read -r -a ranges <<< "${_sr}"
else
  echo "[merge] ERROR: set SHARD_LIST_FILE or SHARD_RANGES" >&2
  exit 1
fi

shard_args=()
for range in "${ranges[@]}"; do
  range="${range// /}"
  [[ -z "${range}" ]] && continue
  start="${range%%:*}"
  end="${range##*:}"
  sp=$(printf "%s.i%03d_%03d.generations.jsonl" "${OUTPUT_JSON%.json}" "${start}" "${end}")
  if [[ ! -f "${sp}" ]]; then
    echo "[merge] ERROR: missing shard generations: ${sp}" >&2
    exit 1
  fi
  shard_args+=(--shard-jsonl "${sp}")
  echo "[merge] shard=${sp}"
done
if [[ ${#shard_args[@]} -eq 0 ]]; then
  echo "[merge] ERROR: no shard ranges resolved" >&2
  exit 1
fi

base_args=()
if [[ -n "${BASE_JSONL}" ]]; then
  if [[ ! -f "${BASE_JSONL}" ]]; then
    echo "[merge] ERROR: missing BASE_JSONL=${BASE_JSONL}" >&2
    exit 1
  fi
  base_args+=(--base-jsonl "${BASE_JSONL}")
  echo "[merge] base=${BASE_JSONL}"
fi

python "${BASE_DIR}/eval_code/merge_lcb_index_shards.py" \
  --data-jsonl "${DATA_JSONL}" \
  --output-json "${OUTPUT_JSON}" \
  --val-n "${VAL_N}" \
  "${base_args[@]}" \
  "${shard_args[@]}"

AUTO_JUDGE=${AUTO_JUDGE:-1}
JUDGE_SCRIPT="${BASE_DIR}/scripts/eval_code/falcon_h1r_7b/lcb_v6_sgl_judge.sh"
# shellcheck source=/dev/null
source "${BASE_DIR}/scripts/eval_code/lcb_auto_submit_judge.sh"
lcb_auto_submit_judge
