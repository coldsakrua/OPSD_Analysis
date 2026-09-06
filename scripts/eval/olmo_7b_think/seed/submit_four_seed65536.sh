#!/bin/bash
# Submit 4 SGLang think evals with SEED=65536. Requires CHECKPOINT_PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export SEED=65536
exec bash "${ROOT}/scripts/eval/olmo_7b_think/seed/submit_four_seed.sh" "$@"
