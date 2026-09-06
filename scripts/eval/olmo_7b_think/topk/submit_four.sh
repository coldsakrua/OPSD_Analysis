#!/bin/bash
# Submit 4 SGLang evals for a top-k KL olmo checkpoint (default SEED=42).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASE_DIR=${BASE_DIR:-${ROOT}}
: "${CHECKPOINT_PATH:?}"
SEED=${SEED:-42}
export SEED
exec bash "${BASE_DIR}/scripts/eval/olmo_7b_think/seed/submit_four_seed.sh"
