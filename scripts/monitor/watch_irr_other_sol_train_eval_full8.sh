#!/bin/bash
set -euo pipefail
BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "${BASE_DIR}"
if [[ -z "${SPECS_OVERRIDE:-}" ]]; then
  SPECS_OVERRIDE=$(cat <<'SPECS'
3096728|qwen3_4b|snt_tnt_1e_6_openthoughts_irr_other_sol|snt_tnt_ios_ckpt100|snt_tnt_ios|4b
3096729|qwen3_1.7b|snt_tnt_1e_6_openthoughts_irr_other_sol_1p7b|snt_tnt_ios_1p7b_ckpt100|snt_tnt_ios_1p7b|1.7b
SPECS
)
  export SPECS_OVERRIDE
fi
export BASE_DIR POLL_SEC="${POLL_SEC:-300}" STEP="${STEP:-100}"
exec bash "${BASE_DIR}/scripts/monitor/watch_multi_train_eval_full8.sh"
