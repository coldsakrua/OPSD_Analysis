#!/bin/bash
# Submit 4 think evals with SEED=65536. Requires CHECKPOINT_PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export SEED=65536
exec bash "${ROOT}/scripts/eval/1.7b/seed/submit_four_seed.sh" "$@"
