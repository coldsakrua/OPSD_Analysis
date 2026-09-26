#!/bin/bash
# Submit all 10 OMR full-solution teacher_prefix jobs.
# Requires shared parquet from scripts/data/prepare_omr_solution_shared2048.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
DS="${ROOT}/data/openmathreasoning/preprocessed/omr.opsd.solution.postthink.shared2048.seed42.maxprompt4096.parquet"
if [[ ! -f "${DS}" ]]; then
  echo "[error] missing ${DS}" >&2
  echo "[error] run prepare first: sbatch scripts/data/prepare_omr_solution_shared2048.sh" >&2
  exit 1
fi
mkdir -p log/data_analysis/22_omr
for s in "$(dirname "${BASH_SOURCE[0]}")"/analyze_*.sh; do
  echo "[submit] $(basename "${s}")"
  sbatch --chdir="${ROOT}" "${s}"
done
echo "[done] submitted OMR full-solution analysis jobs"
