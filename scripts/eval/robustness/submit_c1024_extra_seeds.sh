#!/bin/bash
# Eval existing OpenThoughts c1024 checkpoints at sampling seeds 1024 and 65536.
# Skips Qwen3.5-4B. Does not retrain. Outputs go to eval_outputs/<tag>_seed<SEED>/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

SEEDS=(1024 65536)

# model | submit_four | checkpoint | base eval tag | job prefix
JOBS=(
  "0.6b|scripts/eval/0.6b/submit_four.sh|outputs/qwen3_06b/st_tt_clip005_1e_6_openthoughts_06b/3293794/checkpoint-100|st_tt_clip005_1e6_06b_ckpt100|06b_c1024"
  "4b|scripts/eval/4b/submit_four.sh|outputs/qwen3_4b/st_tt_clip005_1e_6_openthoughts_4b/3219784/checkpoint-100|st_tt_clip005_4b_ckpt100|4b_c1024"
  "4b_instruct|scripts/eval/4b_instruct/submit_four.sh|outputs/qwen3_4b_instruct/snt_tnt_clip005_1e_6_openthoughts_instruct/3213946/checkpoint-100|snt_tnt_clip005_1e6_oti_ckpt100|4bi_c1024"
  "4b_thinking|scripts/eval/qwen3_4b_thinking/submit_four.sh|outputs/qwen3_4b_thinking/st_tt_clip005_1e_6_openthoughts_4bt/3277260/checkpoint-100|st_tt_clip005_1e6_4bt_ckpt100|4bt_c1024"
  "8b|scripts/eval/8b/submit_four.sh|outputs/qwen3_8b/st_tt_clip005_1e_6_openthoughts_8b/3294276/checkpoint-100|st_tt_clip005_1e6_8b_ckpt100|8b_c1024"
  "olmo3_7b_instruct|scripts/eval/olmo3_7b_instruct/submit_four.sh|outputs/olmo3_7b_instruct/snt_tnt_clip005_1e_6_openthoughts_olmo7bit/3213949/checkpoint-100|snt_tnt_clip005_1e6_olmo7bi_ckpt100|olmo7bi_c1024"
  "deepseek_r1_1.5b|scripts/eval/deepseek_r1_1.5b/submit_four.sh|outputs/deepseek_r1_1.5b/st_tt_clip005_1e_6_openthoughts_r1_15b/3297097/checkpoint-100|st_tt_clip005_1e6_r1_15b_ckpt100|r1_c1024"
  "falcon_h1r_7b|scripts/eval/falcon_h1r_7b/submit_four.sh|outputs/falcon_h1r_7b/st_tt_clip005_1e_6_openthoughts_falcon7b/3293706/checkpoint-100|st_tt_clip005_1e6_falcon7b_ckpt100|falcon_c1024"
  "mimo_7b_rl|scripts/eval/mimo_7b_rl/submit_four.sh|outputs/mimo_7b_rl/st_tt_clip005_1e_6_openthoughts_mimo7b/3289222/checkpoint-100|st_tt_clip005_1e6_mimo7b_ckpt100|mimo_c1024"
)

echo "[c1024-seed] root=${ROOT} seeds=${SEEDS[*]} models=${#JOBS[@]} (skip qwen3.5-4b)"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r key submit_four ckpt tag prefix <<<"${entry}"
  ckpt_abs="${ROOT}/${ckpt}"
  if [[ ! -f "${ckpt_abs}/config.json" ]]; then
    echo "[error] missing checkpoint: ${ckpt_abs}" >&2
    exit 1
  fi
  for seed in "${SEEDS[@]}"; do
    echo "[c1024-seed] ${key} seed=${seed} tag=${tag}_seed${seed}"
    CHECKPOINT_PATH="${ckpt_abs}" \
      EVAL_TAG="${tag}_seed${seed}" \
      JOB_PREFIX="${prefix}_s${seed}" \
      SEED="${seed}" \
      BASE_DIR="${ROOT}" \
      bash "${ROOT}/${submit_four}"
  done
done
echo "[c1024-seed] submitted $((${#JOBS[@]} * ${#SEEDS[@]})) models×seeds × 4 datasets"
