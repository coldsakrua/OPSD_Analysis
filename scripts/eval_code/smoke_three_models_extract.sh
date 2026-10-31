#!/bin/bash
# Submit 3 GPU smoke jobs (one per model): each runs LCB / HumanEval+ / MBPP+
# with a few problems and prints code-extraction diagnostics.
#
# Usage (from OPSD_Analysis):
#   bash scripts/eval_code/smoke_three_models_extract.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
mkdir -p log/eval_code/smoke

MAX_PROBLEMS=${MAX_PROBLEMS:-2}
VAL_N=${VAL_N:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-12288}
SEED=${SEED:-42}

chmod +x \
  scripts/eval_code/deepseek_r1_1.5b/smoke_three_extract.sh \
  scripts/eval_code/falcon_h1r_7b/smoke_three_extract.sh \
  scripts/eval_code/mimo_7b_rl/smoke_three_extract.sh \
  eval_code/smoke_extract_report.py

for script in \
  scripts/eval_code/deepseek_r1_1.5b/smoke_three_extract.sh \
  scripts/eval_code/falcon_h1r_7b/smoke_three_extract.sh \
  scripts/eval_code/mimo_7b_rl/smoke_three_extract.sh
do
  jid=$(sbatch --parsable --chdir="${ROOT}" \
    --export=ALL,BASE_DIR="${ROOT}",MAX_PROBLEMS="${MAX_PROBLEMS}",VAL_N="${VAL_N}",MAX_NEW_TOKENS="${MAX_NEW_TOKENS}",SEED="${SEED}" \
    "${ROOT}/${script}")
  echo "[smoke] ${script} -> jid=${jid}"
done
