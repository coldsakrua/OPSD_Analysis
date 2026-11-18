#!/bin/bash
# Eval seeds 0-6 for Qwen3-1.7B position-loss ablation (seed-42 trained ckpts):
#   First-256 ≈ c256 (already has 0-6 via submit_1p7b_c256_c1024_seeds_0_6.sh)
#   Uni-256 / Last-256 (need 0-6; already have 42 / 1024 / 65536)
# Does not retrain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

JOBS=(
  "uni256|outputs/qwen3_1.7b/st_tt_clip005_c1024_uni256_1p7b/3419074/checkpoint-100|st_tt_clip005_c1024_uni256_1p7b_checkpoint-100"
  "last256|outputs/qwen3_1.7b/st_tt_clip005_c1024_last256_1p7b/3419075/checkpoint-100|st_tt_clip005_c1024_last256_1p7b_checkpoint-100"
)

for entry in "${JOBS[@]}"; do
  IFS='|' read -r key ckpt tag <<<"${entry}"
  ckpt_abs="${ROOT}/${ckpt}"
  if [[ ! -f "${ckpt_abs}/config.json" ]]; then
    echo "[error] missing checkpoint: ${ckpt_abs}" >&2
    exit 1
  fi
  for seed in 0 1 2 3 4 5 6; do
    out_dir="${ROOT}/eval_outputs/${tag}_seed${seed}"
    n_ok=$(ls "${out_dir}"/aime24*.metrics.json "${out_dir}"/aime25*.metrics.json \
                 "${out_dir}"/aime26*.metrics.json "${out_dir}"/hmmt25*.metrics.json \
                 2>/dev/null | wc -l || true)
    if [[ "${n_ok}" -ge 4 ]]; then
      echo "[1p7b-pos-seed] skip ${key} seed=${seed} (already have 4 metrics)"
      continue
    fi
    echo "[1p7b-pos-seed] ${key} seed=${seed} tag=${tag}_seed${seed}"
    CHECKPOINT_PATH="${ckpt_abs}" \
      EVAL_TAG="${tag}_seed${seed}" \
      SEED="${seed}" \
      BASE_DIR="${ROOT}" \
      bash "${ROOT}/scripts/eval/1.7b/seed/submit_four_seed.sh"
  done
done
echo "[1p7b-pos-seed] submitted uni256/last256 seeds 0-6 (c256 already covered)"
