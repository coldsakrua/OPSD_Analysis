#!/bin/bash
# Eval seeds 0-6 for Qwen3-1.7B clip005 OpenThoughts c256 and c1024.
# Same seed-42 checkpoints already evaluated at seeds 42, 1024, and 65536.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

# variant | checkpoint | eval tag prefix (seed is appended)
JOBS=(
  "c1024|outputs/qwen3_1.7b/st_tt_clip005_1e_6_openthoughts_1p7b/3212996/checkpoint-100|st_tt_clip005_1e_6_openthoughts_1p7b_checkpoint-100"
  "c256|outputs/qwen3_1.7b/st_tt_clip005_1e_6_ot_1p7b_c256/3226382/checkpoint-100|st_tt_clip005_1e_6_ot_1p7b_c256_checkpoint-100"
)

for entry in "${JOBS[@]}"; do
  IFS='|' read -r key ckpt tag <<<"${entry}"
  ckpt_abs="${ROOT}/${ckpt}"
  if [[ ! -f "${ckpt_abs}/config.json" ]]; then
    echo "[error] missing checkpoint: ${ckpt_abs}" >&2
    exit 1
  fi
  for seed in 0 1 2 3 4 5 6; do
    echo "[1p7b-ot-seed] ${key} seed=${seed} tag=${tag}_seed${seed}"
    CHECKPOINT_PATH="${ckpt_abs}" \
      EVAL_TAG="${tag}_seed${seed}" \
      SEED="${seed}" \
      BASE_DIR="${ROOT}" \
      bash "${ROOT}/scripts/eval/1.7b/seed/submit_four_seed.sh"
  done
done
echo "[1p7b-ot-seed] submitted c256 and c1024 seeds 0-6"
