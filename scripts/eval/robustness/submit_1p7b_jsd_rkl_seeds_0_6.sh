#!/bin/bash
# Eval seeds 0-6 for Qwen3-1.7B three-KL ablation:
#   forward KL = st_tt_clip005 (already covered by submit_1p7b_c256_c1024_seeds_0_6.sh)
#   JSD        = st_tt_jsd_clip005 (beta=0.5)
#   reverse KL = st_tt_rkl_clip005 (beta=1.0)
# Existing seeds 42 / 1024 / 65536 are left in place.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

# variant | checkpoint | eval tag prefix (seed is appended)
JOBS=(
  "jsd|outputs/qwen3_1.7b/st_tt_jsd_clip005_1e_6_openthoughts_1p7b/3378663/checkpoint-100|st_tt_jsd_clip005_1e_6_openthoughts_1p7b_ckpt100"
  "rkl|outputs/qwen3_1.7b/st_tt_rkl_clip005_1e_6_openthoughts_1p7b/3378662/checkpoint-100|st_tt_rkl_clip005_1e_6_openthoughts_1p7b_ckpt100"
)

for entry in "${JOBS[@]}"; do
  IFS='|' read -r key ckpt tag <<<"${entry}"
  ckpt_abs="${ROOT}/${ckpt}"
  if [[ ! -f "${ckpt_abs}/config.json" ]]; then
    echo "[error] missing checkpoint: ${ckpt_abs}" >&2
    exit 1
  fi
  for seed in 0 1 2 3 4 5 6; do
    echo "[1p7b-kl-seed] ${key} seed=${seed} tag=${tag}_seed${seed}"
    CHECKPOINT_PATH="${ckpt_abs}" \
      EVAL_TAG="${tag}_seed${seed}" \
      SEED="${seed}" \
      BASE_DIR="${ROOT}" \
      bash "${ROOT}/scripts/eval/1.7b/seed/submit_four_seed.sh"
  done
done
echo "[1p7b-kl-seed] submitted JSD and reverse-KL seeds 0-6"
