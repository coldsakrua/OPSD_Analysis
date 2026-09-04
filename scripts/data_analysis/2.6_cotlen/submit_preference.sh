#!/bin/bash
# Submit preference / data-analysis only for an existing cotlen output dir
# (must already contain rollouts.jsonl; accuracy_*.json optional).
#
# Usage:
#   ./submit_preference.sh scripts/data_analysis/outputs/cotlen/qwen3_4b_instruct/snt_tnt_easy_3421149
#   ./submit_preference.sh <output_dir> analyze_snt_tnt_easy_qwen3_4b_instruct.sh
set -euo pipefail
BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
DIR="${BASE_DIR}/scripts/data_analysis/2.6_cotlen"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <OUTPUT_DIR> [analyze_*.sh]" >&2
  exit 1
fi

OUTPUT_DIR=$(readlink -f "$1")
if [[ ! -f "${OUTPUT_DIR}/rollouts.jsonl" ]]; then
  echo "missing rollouts.jsonl under ${OUTPUT_DIR}" >&2
  exit 1
fi

SCRIPT_NAME=${2:-}
if [[ -z "${SCRIPT_NAME}" ]]; then
  # Infer from path: .../cotlen/<model_key>/<combo>_<band>_<jobid>
  model_key=$(basename "$(dirname "${OUTPUT_DIR}")")
  leaf=$(basename "${OUTPUT_DIR}")
  # leaf e.g. snt_tnt_easy_3421149 or st_tt_hard_3421171
  if [[ "${leaf}" =~ ^(st_tt|snt_tnt)_(easy|hard)_ ]]; then
    combo="${BASH_REMATCH[1]}"
    band="${BASH_REMATCH[2]}"
  else
    echo "cannot infer analyze script from ${leaf}; pass it as arg2" >&2
    exit 1
  fi
  SCRIPT_NAME="analyze_${combo}_${band}_${model_key}.sh"
fi

SCRIPT="${DIR}/${SCRIPT_NAME}"
if [[ ! -f "${SCRIPT}" ]]; then
  echo "missing script: ${SCRIPT}" >&2
  exit 1
fi

echo "[submit preference] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[submit preference] script=${SCRIPT}"
# Preference scoring is CPU-RAM heavy; cluster caps ~112G for 7 cpus/gpu.
PREF_MEM=${PREF_MEM:-110G}
cd "${BASE_DIR}"
sbatch --mem="${PREF_MEM}" \
  --export=ALL,SKIP_GENERATE=1,RUN_ACCURACY=0,RUN_PREFERENCE=1,OUTPUT_DIR="${OUTPUT_DIR}" \
  "${SCRIPT}"
