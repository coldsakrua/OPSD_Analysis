#!/bin/bash
# Extra sampling seeds for the aligned answer-only and unrelated checkpoints.
# Seed 42 was already submitted by the watcher. This adds 1024 and 65536.
# Outputs: eval_outputs/<eval_tag>_seed<SEED>/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

SEEDS=(1024 65536)

# key | submit_four | checkpoint | base eval tag | job prefix
JOBS=(
  "1.7b-ans|scripts/eval/1.7b/submit_four.sh|outputs/qwen3_1.7b/st_tt_clip005_1e_6_ot_ans_same6400_1p7b/5066957/checkpoint-100|st_tt_ot_ans_same6400_1p7b_ckpt100|17b_ans"
  "1.7b-ios|scripts/eval/1.7b/submit_four.sh|outputs/qwen3_1.7b/st_tt_clip005_1e_6_ot_ios_same6400_1p7b/5066958/checkpoint-100|st_tt_ot_ios_same6400_1p7b_ckpt100|17b_ios"
  "4b-ans|scripts/eval/4b/submit_four.sh|outputs/qwen3_4b/st_tt_clip005_1e_6_ot_ans_same3200_4b/5066959/checkpoint-100|st_tt_ot_ans_same3200_4b_ckpt100|4b_ans"
  "4b-ios|scripts/eval/4b/submit_four.sh|outputs/qwen3_4b/st_tt_clip005_1e_6_ot_ios_same3200_4b/5066960/checkpoint-100|st_tt_ot_ios_same3200_4b_ckpt100|4b_ios"
  "4bi-ans|scripts/eval/4b_instruct/submit_four.sh|outputs/qwen3_4b_instruct/snt_tnt_clip005_1e_6_ot_ans_same3200_instruct/5066961/checkpoint-100|snt_tnt_ot_ans_same3200_instruct_ckpt100|4bi_ans"
  "4bi-ios|scripts/eval/4b_instruct/submit_four.sh|outputs/qwen3_4b_instruct/snt_tnt_clip005_1e_6_ot_ios_same3200_instruct/5066962/checkpoint-100|snt_tnt_ot_ios_same3200_instruct_ckpt100|4bi_ios"
  "olmo7bt-ans|scripts/eval/olmo_7b_think/submit_four.sh|outputs/olmo3_7b_think/st_tt_clip005_1e_6_ot_ans_same6400_olmo7bt/5066963/checkpoint-100|st_tt_ot_ans_same6400_olmo7bt_ckpt100|olmo7bt_ans"
  "olmo7bt-ios|scripts/eval/olmo_7b_think/submit_four.sh|outputs/olmo3_7b_think/st_tt_clip005_1e_6_ot_ios_same6400_olmo7bt/5066964/checkpoint-100|st_tt_ot_ios_same6400_olmo7bt_ckpt100|olmo7bt_ios"
  "olmo7bi-ans|scripts/eval/olmo3_7b_instruct/submit_four.sh|outputs/olmo3_7b_instruct/snt_tnt_clip005_1e_6_ot_ans_same6400_olmo7bit/5066965/checkpoint-100|snt_tnt_ot_ans_same6400_olmo7bit_ckpt100|olmo7bi_ans"
  "olmo7bi-ios|scripts/eval/olmo3_7b_instruct/submit_four.sh|outputs/olmo3_7b_instruct/snt_tnt_clip005_1e_6_ot_ios_same6400_olmo7bit/5066966/checkpoint-100|snt_tnt_ot_ios_same6400_olmo7bit_ckpt100|olmo7bi_ios"
)

echo "[aligned-seed] root=${ROOT} seeds=${SEEDS[*]} runs=${#JOBS[@]}"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r key submit_four ckpt tag prefix <<<"${entry}"
  ckpt_abs="${ROOT}/${ckpt}"
  if [[ ! -f "${ckpt_abs}/config.json" ]]; then
    echo "[error] missing checkpoint: ${ckpt_abs}" >&2
    exit 1
  fi
  for seed in "${SEEDS[@]}"; do
    echo "[aligned-seed] ${key} seed=${seed} tag=${tag}_seed${seed}"
    CHECKPOINT_PATH="${ckpt_abs}" \
      EVAL_TAG="${tag}_seed${seed}" \
      JOB_PREFIX="${prefix}_s${seed}" \
      SEED="${seed}" \
      BASE_DIR="${ROOT}" \
      bash "${ROOT}/${submit_four}"
  done
done
echo "[aligned-seed] submitted $((${#JOBS[@]} * ${#SEEDS[@]})) runs×seeds × 4 datasets"
