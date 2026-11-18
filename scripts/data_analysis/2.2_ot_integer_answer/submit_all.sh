#!/bin/bash
# Submit OT integer answer-privilege analysis for qwen3_1.7b + olmo3_7b_think.
# Requires shared2048 packs from scripts/data/prepare_ot_integer_answer_variants_shared2048.py
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
PRE="${ROOT}/data/openthoughts/preprocessed"
for ds in \
  "${PRE}/openthoughts.correct.answer.sthink_tthink.integer_answer.shared2048.seed42.maxprompt1024.parquet" \
  "${PRE}/openthoughts.correct.answer.sthink_tthink.integer_answer.rewritten_frac_decimal.shared2048.seed42.maxprompt1024.parquet" \
  "${PRE}/openthoughts.correct.answer.sthink_tthink.integer_answer.wrong_n10p1.shared2048.seed42.maxprompt1024.parquet" \
  "${PRE}/openthoughts.correct.answer.sthink_tthink.integer_answer.wrong_rand50.shared2048.seed42.maxprompt1024.parquet"
do
  if [[ ! -f "${ds}" ]]; then
    echo "[error] missing ${ds}" >&2
    echo "[error] run: python scripts/data/prepare_ot_integer_answer_variants_shared2048.py" >&2
    exit 1
  fi
done
mkdir -p log/data_analysis/22_ot_integer_answer
for s in \
  "$(dirname "${BASH_SOURCE[0]}")/analyze_st_tt_qwen3_1_7b.sh" \
  "$(dirname "${BASH_SOURCE[0]}")/analyze_st_tt_olmo3_7b_think.sh"
do
  echo "[submit] $(basename "${s}")"
  sbatch --chdir="${ROOT}" "${s}"
done
echo "[done] submitted OT integer answer-privilege analysis jobs"
