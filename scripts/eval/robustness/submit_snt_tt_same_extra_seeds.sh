#!/bin/bash
# Eval snt_tt same (no-privilege) checkpoints at sampling seeds 1024 and 65536.
# 3 ckpts × 2 seeds × 4 datasets. Does not retrain. Seed-42 results stay in place.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

SEEDS=(1024 65536)

# model | submit_four | checkpoint | base eval tag | job prefix
JOBS=(
  "1.7b|scripts/eval/1.7b/submit_four.sh|outputs/qwen3_1.7b/snt_tt_same_clip005_1e_6_openthoughts_1p7b/3749335/checkpoint-100|snt_tt_same_clip005_1p7b_ckpt100|sntsame_1p7b"
  "4b|scripts/eval/4b/submit_four.sh|outputs/qwen3_4b/snt_tt_same_clip005_1e_6_openthoughts_4b/3749336/checkpoint-100|snt_tt_same_clip005_4b_ckpt100|sntsame_4b"
  "8b|scripts/eval/8b/submit_four.sh|outputs/qwen3_8b/snt_tt_same_clip005_1e_6_openthoughts_8b/3749337/checkpoint-100|snt_tt_same_clip005_8b_ckpt100|sntsame_8b"
)

echo "[snt-tt-same-seed] root=${ROOT} seeds=${SEEDS[*]} ckpts=${#JOBS[@]}"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r key submit_four ckpt tag prefix <<<"${entry}"
  ckpt_abs="${ROOT}/${ckpt}"
  if [[ ! -f "${ckpt_abs}/config.json" ]]; then
    echo "[error] missing checkpoint: ${ckpt_abs}" >&2
    exit 1
  fi
  for seed in "${SEEDS[@]}"; do
    echo "[snt-tt-same-seed] ${key} seed=${seed} tag=${tag}_seed${seed}"
    CHECKPOINT_PATH="${ckpt_abs}" \
      EVAL_TAG="${tag}_seed${seed}" \
      JOB_PREFIX="${prefix}_s${seed}" \
      SEED="${seed}" \
      BASE_DIR="${ROOT}" \
      bash "${ROOT}/${submit_four}"
  done
done
echo "[snt-tt-same-seed] submitted $((${#JOBS[@]} * ${#SEEDS[@]})) ckpt×seed groups × 4 datasets (=24 jobs)"
