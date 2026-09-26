#!/bin/bash
# Submit A40 variants of the currently pending DAPO answer-only teacher_prefix jobs.
# Original A800 jobs are left untouched.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
DS="${ROOT}/data/dapo/preprocessed/dapo-math-17k.answer_only.shared2048.seed42.maxprompt1024.parquet"
if [[ ! -f "${DS}" ]]; then
  echo "[error] missing ${DS}" >&2
  echo "[error] run prepare first: sbatch scripts/data/prepare_dapo_answer_shared2048.sh" >&2
  exit 1
fi
mkdir -p log/data_analysis/22_dapo
for s in "$(dirname "${BASH_SOURCE[0]}")"/analyze_*_a40.sh; do
  echo "[submit] $(basename "${s}")"
  sbatch --chdir="${ROOT}" "${s}"
done
echo "[done] submitted pending A40 DAPO answer-only analysis jobs"
