#!/bin/bash
# Submit 2.6 cotlen jobs that have preprocessed parquet ready.
# Each job: generate + boxed accuracy only (preference is separate: submit_preference.sh).
# Skip qwen3_8b until its cotlen parquet is built.
set -euo pipefail
BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
DIR="${BASE_DIR}/scripts/data_analysis/2.6_cotlen"

SCRIPTS=(
  analyze_st_tt_easy_qwen3_1.7b.sh
  analyze_st_tt_hard_qwen3_1.7b.sh
  analyze_st_tt_easy_qwen3_4b.sh
  analyze_st_tt_hard_qwen3_4b.sh
  analyze_snt_tnt_easy_qwen3_4b_instruct.sh
  analyze_snt_tnt_hard_qwen3_4b_instruct.sh
  analyze_st_tt_easy_olmo3_7b_think.sh
  analyze_st_tt_hard_olmo3_7b_think.sh
  analyze_snt_tnt_easy_olmo3_7b_instruct.sh
  analyze_snt_tnt_hard_olmo3_7b_instruct.sh
)

cd "${BASE_DIR}"
for s in "${SCRIPTS[@]}"; do
  echo "[submit] ${s}"
  sbatch "${DIR}/${s}"
done
