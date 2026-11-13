#!/bin/bash
# Extra eval seeds for Qwen3-1.7B same-order answer-only / unrelated c256.
# Watcher already submitted seed 42. This adds 0-6, 1024, and 65536 to match
# full-solution c256 coverage. Outputs: eval_outputs/<tag>_seed<SEED>/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

SEEDS=(0 1 2 3 4 5 6 1024 65536)

# key | submit_script | checkpoint | base eval tag | job prefix
JOBS=(
  "1.7b-ans|scripts/eval/1.7b/seed/submit_four_seed.sh|outputs/qwen3_1.7b/st_tt_clip005_1e_6_ot_ans_same6400_1p7b_c256/5113621/checkpoint-100|st_tt_ot_ans_same6400_1p7b_c256_ckpt100|17b_ans"
  "1.7b-ios|scripts/eval/1.7b/seed/submit_four_seed.sh|outputs/qwen3_1.7b/st_tt_clip005_1e_6_ot_ios_same6400_1p7b_c256/5113622/checkpoint-100|st_tt_ot_ios_same6400_1p7b_c256_ckpt100|17b_ios"
)

echo "[same-c256-seed] root=${ROOT} seeds=${SEEDS[*]} runs=${#JOBS[@]}"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r key submit_script ckpt tag prefix <<<"${entry}"
  ckpt_abs="${ROOT}/${ckpt}"
  if [[ ! -f "${ckpt_abs}/config.json" ]]; then
    echo "[error] missing checkpoint: ${ckpt_abs}" >&2
    exit 1
  fi
  for seed in "${SEEDS[@]}"; do
    echo "[same-c256-seed] ${key} seed=${seed} tag=${tag}_seed${seed}"
    CHECKPOINT_PATH="${ckpt_abs}" \
      EVAL_TAG="${tag}_seed${seed}" \
      JOB_PREFIX="${prefix}_s${seed}" \
      SEED="${seed}" \
      BASE_DIR="${ROOT}" \
      bash "${ROOT}/${submit_script}"
  done
done
echo "[same-c256-seed] submitted $((${#JOBS[@]} * ${#SEEDS[@]})) runs×seeds"
