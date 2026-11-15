#!/bin/bash
# Eval Table-6 50-step OPSD / GRPO checkpoints at sampling seeds 1024 and 65536.
# 6 ckpts × 2 seeds × 4 datasets. Does not retrain. Seed-42 results stay in place.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

SEEDS=(1024 65536)

# model | submit_four | checkpoint | base eval tag | job prefix
JOBS=(
  "1.7b_opsd|scripts/eval/1.7b/submit_four.sh|outputs/qwen3_1.7b/st_tt_clip005_1e_6_omr_solution_postthink_50step_1p7b/3857927/checkpoint-50|st_tt_clip005_1e_6_omr_solution_postthink_50step_1p7b_ckpt50|opsd50_1p7b"
  "4b_opsd|scripts/eval/4b/submit_four.sh|outputs/qwen3_4b/st_tt_clip005_1e_6_omr_solution_postthink_50step_4b/3857928/checkpoint-50|st_tt_clip005_1e_6_omr_solution_postthink_50step_4b_ckpt50|opsd50_4b"
  "8b_opsd|scripts/eval/8b/submit_four.sh|outputs/qwen3_8b/st_tt_clip005_1e_6_omr_solution_postthink_50step_8b/4113380/checkpoint-50|st_tt_clip005_1e_6_omr_solution_postthink_50step_8b_ckpt50|opsd50_8b"
  "1.7b_grpo|scripts/eval/1.7b/submit_four.sh|outputs/qwen3_1.7b/grpo_think_1p7b_omr_int_4gpu_bs64_24k_6400/3678602/global_step_50/actor/hf_merged|grpo_omr_int_1p7b_bs64_24k_gs100|grpo50_1p7b"
  "4b_grpo|scripts/eval/4b/submit_four.sh|outputs/qwen3_4b/grpo_think_4b_omr_int_8gpu_bs64_24k_6400/3719725/global_step_50/actor/hf_merged|grpo_omr_int_4b_8gpu_bs64_24k_gs50|grpo50_4b"
  "8b_grpo|scripts/eval/8b/submit_four.sh|outputs/qwen3_8b/grpo_think_8b_omr_int_4gpu_bs64_24k_3200/4045322/global_step_50/actor/hf_merged|grpo_think_8b_omr_int_4gpu_bs64_24k_3200_gs50|grpo50_8b"
)

echo "[table6-50step-seed] root=${ROOT} seeds=${SEEDS[*]} ckpts=${#JOBS[@]}"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r key submit_four ckpt tag prefix <<<"${entry}"
  ckpt_abs="${ROOT}/${ckpt}"
  if [[ ! -f "${ckpt_abs}/config.json" ]]; then
    echo "[error] missing checkpoint: ${ckpt_abs}" >&2
    exit 1
  fi
  for seed in "${SEEDS[@]}"; do
    echo "[table6-50step-seed] ${key} seed=${seed} tag=${tag}_seed${seed}"
    CHECKPOINT_PATH="${ckpt_abs}" \
      EVAL_TAG="${tag}_seed${seed}" \
      JOB_PREFIX="${prefix}_s${seed}" \
      SEED="${seed}" \
      BASE_DIR="${ROOT}" \
      bash "${ROOT}/${submit_four}"
  done
done
echo "[table6-50step-seed] submitted $((${#JOBS[@]} * ${#SEEDS[@]})) ckpt×seed groups × 4 datasets (=48 jobs)"
