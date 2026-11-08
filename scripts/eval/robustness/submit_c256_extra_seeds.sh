#!/bin/bash
# Eval existing OpenThoughts c256 checkpoints at sampling seeds 1024 and 65536.
# Same 9 models as submit_c1024_extra_seeds.sh. Does not retrain.
# Outputs go to eval_outputs/<tag>_seed<SEED>/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

SEEDS=(1024 65536)

# model | submit_four | checkpoint | base eval tag | job prefix
JOBS=(
  "0.6b|scripts/eval/0.6b/submit_four.sh|outputs/qwen3_06b/st_tt_clip005_1e_6_ot_06b_c256/3293795/checkpoint-100|st_tt_clip005_c256_06b_ckpt100|06b_c256"
  "4b|scripts/eval/4b/submit_four.sh|outputs/qwen3_4b/st_tt_clip005_1e_6_ot_4b_c256/3226380/checkpoint-100|st_tt_clip005_4b_c256_ckpt100|4b_c256"
  "4b_instruct|scripts/eval/4b_instruct/submit_four.sh|outputs/qwen3_4b_instruct/snt_tnt_clip005_1e_6_oti_c256/3226358/checkpoint-100|snt_tnt_clip005_4bi_c256_ckpt100|4bi_c256"
  "4b_thinking|scripts/eval/qwen3_4b_thinking/submit_four.sh|outputs/qwen3_4b_thinking/st_tt_clip005_1e_6_ot_4bt_c256/3277261/checkpoint-100|st_tt_clip005_c256_4bt_ckpt100|4bt_c256"
  "8b|scripts/eval/8b/submit_four.sh|outputs/qwen3_8b/st_tt_clip005_1e_6_ot_8b_c256/3346472/checkpoint-100|st_tt_clip005_1e6_8b_c256_ckpt100|8b_c256"
  "olmo3_7b_instruct|scripts/eval/olmo3_7b_instruct/submit_four.sh|outputs/olmo3_7b_instruct/snt_tnt_clip005_1e_6_oti_olmo7bi_c256/3226917/checkpoint-100|snt_tnt_clip005_olmo7bi_c256_ckpt100|olmo7bi_c256"
  "deepseek_r1_1.5b|scripts/eval/deepseek_r1_1.5b/submit_four.sh|outputs/deepseek_r1_1.5b/st_tt_clip005_1e_6_ot_r1_15b_c256/3297098/checkpoint-100|st_tt_clip005_c256_r1_15b_ckpt100|r1_c256"
  "falcon_h1r_7b|scripts/eval/falcon_h1r_7b/submit_four.sh|outputs/falcon_h1r_7b/st_tt_clip005_1e_6_ot_falcon7b_c256/3293707/checkpoint-100|st_tt_clip005_c256_falcon7b_ckpt100|falcon_c256"
  "mimo_7b_rl|scripts/eval/mimo_7b_rl/submit_four.sh|outputs/mimo_7b_rl/st_tt_clip005_1e_6_ot_mimo7b_c256/3289223/checkpoint-100|st_tt_clip005_c256_mimo7b_ckpt100|mimo_c256"
)

echo "[c256-seed] root=${ROOT} seeds=${SEEDS[*]} models=${#JOBS[@]}"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r key submit_four ckpt tag prefix <<<"${entry}"
  ckpt_abs="${ROOT}/${ckpt}"
  if [[ ! -f "${ckpt_abs}/config.json" ]]; then
    echo "[error] missing checkpoint: ${ckpt_abs}" >&2
    exit 1
  fi
  for seed in "${SEEDS[@]}"; do
    echo "[c256-seed] ${key} seed=${seed} tag=${tag}_seed${seed}"
    CHECKPOINT_PATH="${ckpt_abs}" \
      EVAL_TAG="${tag}_seed${seed}" \
      JOB_PREFIX="${prefix}_s${seed}" \
      SEED="${seed}" \
      BASE_DIR="${ROOT}" \
      bash "${ROOT}/${submit_four}"
  done
done
echo "[c256-seed] submitted $((${#JOBS[@]} * ${#SEEDS[@]})) models×seeds × 4 datasets"
