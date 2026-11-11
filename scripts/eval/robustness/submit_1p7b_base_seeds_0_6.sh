#!/bin/bash
# Extra eval seeds 0-6 for the Qwen3-1.7B model itself (not qwen3-1.7b-base).
# Existing seeds 42 / 1024 / 65536 are left in place.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

CKPT="/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b"
if [[ ! -f "${CKPT}/config.json" ]]; then
  echo "[error] missing checkpoint: ${CKPT}" >&2
  exit 1
fi

for seed in 0 1 2 3 4 5 6; do
  echo "[1p7b-base-seed] seed=${seed} tag=qwen3-1.7b_seed${seed}"
  CHECKPOINT_PATH="${CKPT}" \
    EVAL_TAG="qwen3-1.7b_seed${seed}" \
    SEED="${seed}" \
    BASE_DIR="${ROOT}" \
    bash "${ROOT}/scripts/eval/1.7b/seed/submit_four_seed.sh"
done
echo "[1p7b-base-seed] submitted seeds 0-6"
