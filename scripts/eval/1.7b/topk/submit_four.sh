#!/bin/bash
# Submit 4 think evals for a top-k KL 1.7b checkpoint (default SEED=42).
# Required: CHECKPOINT_PATH
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BASE_DIR=${BASE_DIR:-${ROOT}}
: "${CHECKPOINT_PATH:?}"
SEED=${SEED:-42}
export SEED
exec bash "${BASE_DIR}/scripts/eval/1.7b/seed/submit_four_seed.sh"
