#!/bin/bash
# Eval seeds 0-6 for Olmo-3-7B-Think position-loss ablation (seed-42 trained ckpts):
#   First-256 ≈ c256, Uni-256, Last-256
# Existing seeds 42 / 1024 / 65536 are left in place. Does not retrain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

JOBS=(
  "c256|outputs/olmo3_7b_think/st_tt_clip005_1e_6_ot_olmo7bt_c256/3236843/checkpoint-100|st_tt_clip005_1e_6_ot_olmo7bt_c256_checkpoint-100"
  "uni256|outputs/olmo3_7b_think/st_tt_clip005_c1024_uni256_olmo7bt/3574089/checkpoint-100|st_tt_clip005_c1024_uni256_olmo7bt_checkpoint-100"
  "last256|outputs/olmo3_7b_think/st_tt_clip005_c1024_last256_olmo7bt/3574090/checkpoint-100|st_tt_clip005_c1024_last256_olmo7bt_checkpoint-100"
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
      echo "[olmo-pos-seed] skip ${key} seed=${seed} (already have 4 metrics)"
      continue
    fi
    echo "[olmo-pos-seed] ${key} seed=${seed} tag=${tag}_seed${seed}"
    CHECKPOINT_PATH="${ckpt_abs}" \
      EVAL_TAG="${tag}_seed${seed}" \
      SEED="${seed}" \
      BASE_DIR="${ROOT}" \
      bash "${ROOT}/scripts/eval/olmo_7b_think/seed/submit_four_seed.sh"
  done
done
echo "[olmo-pos-seed] submitted c256/uni256/last256 seeds 0-6"
