#!/bin/bash
#SBATCH --job-name=lcb_falcon7b_merge
#SBATCH --output=log/eval_code/falcon_h1r_7b/lcb_v6/sgl/%x.%j.out
#SBATCH --partition=C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#
# Merge 4× n=1 shard generations, then submit CPU judge.
# Required env: BASE_DIR OUTPUT_JSON DATA_JSONL
# Optional: N1_SHARDS=4
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
N1_SHARDS=${N1_SHARDS:-4}

cd "${BASE_DIR}"
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval_code/falcon_h1r_7b/lcb_v6/sgl"

shard_args=()
for i in $(seq 0 $((N1_SHARDS - 1))); do
  sp="${OUTPUT_JSON%.json}.n1_s${i}.generations.json"
  if [[ ! -f "${sp}" ]]; then
    echo "[merge] ERROR: missing shard generations: ${sp}" >&2
    exit 1
  fi
  shard_args+=(--shard-json "${sp}")
  echo "[merge] shard${i}=${sp}"
done

python "${BASE_DIR}/eval_code/merge_lcb_n1_shards.py" \
  "${shard_args[@]}" \
  --output-json "${OUTPUT_JSON}"

AUTO_JUDGE=${AUTO_JUDGE:-1}
JUDGE_SCRIPT="${BASE_DIR}/scripts/eval_code/falcon_h1r_7b/lcb_v6_sgl_judge.sh"
# shellcheck source=/dev/null
source "${BASE_DIR}/scripts/eval_code/lcb_auto_submit_judge.sh"
lcb_auto_submit_judge
