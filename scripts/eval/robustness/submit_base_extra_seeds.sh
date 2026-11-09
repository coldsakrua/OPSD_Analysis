#!/bin/bash
# Eval the nine pretrained base models at sampling seeds 1024 and 65536.
# Qwen3-1.7B and OLMo3-7B-Think already have these. Qwen3.5-4B is skipped.
# Does not overwrite the seed-42 eval directories.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

SEEDS=(1024 65536)

# model | submit_four | checkpoint | base eval tag | job prefix
JOBS=(
  "0.6b|scripts/eval/0.6b/submit_four.sh|/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-0.6b|qwen3-0.6b|06b_base"
  "4b|scripts/eval/4b/submit_four.sh|/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b|qwen3-4b|4b_base"
  "4b_instruct|scripts/eval/4b_instruct/submit_four.sh|/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b-instruct|qwen3-4b-instruct|4bi_base"
  "4b_thinking|scripts/eval/qwen3_4b_thinking/submit_four.sh|/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b-thinking|qwen3-4b-thinking|4bt_base"
  "8b|scripts/eval/8b/submit_four.sh|/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-8b|qwen3-8b|8b_base"
  "olmo3_7b_instruct|scripts/eval/olmo3_7b_instruct/submit_four.sh|/dfs/share/maleiLab/2501210611/model/olmo3-7b-it|olmo3-7b-it|olmo7bi_base"
  "deepseek_r1_1.5b|scripts/eval/deepseek_r1_1.5b/submit_four.sh|/gpfs/share/home/2501210611/labShare/2501210611/model/deepseek-r1-distill-qwen-1.5b|deepseek-r1-distill-qwen-1.5b|r1_base"
  "falcon_h1r_7b|scripts/eval/falcon_h1r_7b/submit_four.sh|/dfs/share/maleiLab/2501210611/model/falcon-h1r-7b|falcon-h1r-7b|falcon_base"
  "mimo_7b_rl|scripts/eval/mimo_7b_rl/submit_four.sh|/dfs/share/maleiLab/2501210611/model/mimo-7b-rl|mimo-7b-rl|mimo_base"
)

echo "[base-seed] root=${ROOT} seeds=${SEEDS[*]} models=${#JOBS[@]}"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r key submit_four ckpt tag prefix <<<"${entry}"
  if [[ ! -f "${ckpt}/config.json" ]]; then
    echo "[error] missing checkpoint: ${ckpt}" >&2
    exit 1
  fi
  for seed in "${SEEDS[@]}"; do
    echo "[base-seed] ${key} seed=${seed} tag=${tag}_seed${seed}"
    CHECKPOINT_PATH="${ckpt}" \
      EVAL_TAG="${tag}_seed${seed}" \
      JOB_PREFIX="${prefix}_s${seed}" \
      SEED="${seed}" \
      BASE_DIR="${ROOT}" \
      bash "${ROOT}/${submit_four}"
  done
done
echo "[base-seed] submitted $((${#JOBS[@]} * ${#SEEDS[@]})) models×seeds × 4 datasets"
