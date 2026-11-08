#!/bin/bash
# Eval seed-42 JSD (beta=0.5) and reverse-KL (beta=1) checkpoints at sampling
# seeds 1024 and 65536. Checkpoints are the ones that reproduce table (a).
# Does not retrain and does not overwrite the seed-42 eval directories.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

SEEDS=(1024 65536)

# model | submit_four | checkpoint | base eval tag | job prefix
JOBS=(
  "1.7b-jsd|scripts/eval/1.7b/submit_four.sh|outputs/qwen3_1.7b/st_tt_jsd_clip005_1e_6_openthoughts_1p7b/3378663/checkpoint-100|st_tt_jsd_clip005_1e_6_openthoughts_1p7b_ckpt100|17b_jsd"
  "1.7b-rkl|scripts/eval/1.7b/submit_four.sh|outputs/qwen3_1.7b/st_tt_rkl_clip005_1e_6_openthoughts_1p7b/3378662/checkpoint-100|st_tt_rkl_clip005_1e_6_openthoughts_1p7b_ckpt100|17b_rkl"
  "4b-jsd|scripts/eval/4b/submit_four.sh|outputs/qwen3_4b/st_tt_jsd_clip005_1e_6_openthoughts_4b/3378665/checkpoint-100|st_tt_jsd_clip005_1e_6_openthoughts_4b_ckpt100|4b_jsd"
  "4b-rkl|scripts/eval/4b/submit_four.sh|outputs/qwen3_4b/st_tt_rkl_clip005_1e_6_openthoughts_4b/3378664/checkpoint-100|st_tt_rkl_clip005_1e_6_openthoughts_4b_ckpt100|4b_rkl"
  "4bt-jsd|scripts/eval/qwen3_4b_thinking/submit_four.sh|outputs/qwen3_4b_thinking/st_tt_jsd_clip005_1e_6_openthoughts_4bt/3378667/checkpoint-100|st_tt_jsd_clip005_1e_6_openthoughts_4bt_ckpt100|4bt_jsd"
  "4bt-rkl|scripts/eval/qwen3_4b_thinking/submit_four.sh|outputs/qwen3_4b_thinking/st_tt_rkl_clip005_1e_6_openthoughts_4bt/3378666/checkpoint-100|st_tt_rkl_clip005_1e_6_openthoughts_4bt_ckpt100|4bt_rkl"
  "olmo7bt-jsd|scripts/eval/olmo_7b_think/submit_four.sh|outputs/olmo3_7b_think/st_tt_jsd_clip005_1e_6_openthoughts_olmo7bt/3384206/checkpoint-100|st_tt_jsd_clip005_1e_6_openthoughts_olmo7bt_ckpt100|olmo7bt_jsd"
  "olmo7bt-rkl|scripts/eval/olmo_7b_think/submit_four.sh|outputs/olmo3_7b_think/st_tt_rkl_clip005_1e_6_openthoughts_olmo7bt/3378668/checkpoint-100|st_tt_rkl_clip005_1e_6_openthoughts_olmo7bt_ckpt100|olmo7bt_rkl"
  "olmo7bi-jsd|scripts/eval/olmo3_7b_instruct/submit_four.sh|outputs/olmo3_7b_instruct/snt_tnt_jsd_clip005_1e_6_openthoughts_olmo7bi/3378671/checkpoint-100|snt_tnt_jsd_clip005_1e_6_openthoughts_olmo7bi_ckpt100|olmo7bi_jsd"
  "olmo7bi-rkl|scripts/eval/olmo3_7b_instruct/submit_four.sh|outputs/olmo3_7b_instruct/snt_tnt_rkl_clip005_1e_6_openthoughts_olmo7bi/3378670/checkpoint-100|snt_tnt_rkl_clip005_1e_6_openthoughts_olmo7bi_ckpt100|olmo7bi_rkl"
  "falcon-jsd|scripts/eval/falcon_h1r_7b/submit_four.sh|outputs/falcon_h1r_7b/st_tt_jsd_clip005_1e_6_openthoughts_falcon7b/3387461/checkpoint-100|st_tt_jsd_clip005_1e_6_openthoughts_falcon7b_ckpt100|falcon_jsd"
  "falcon-rkl|scripts/eval/falcon_h1r_7b/submit_four.sh|outputs/falcon_h1r_7b/st_tt_rkl_clip005_1e_6_openthoughts_falcon7b/3387462/checkpoint-100|st_tt_rkl_clip005_1e_6_openthoughts_falcon7b_ckpt100|falcon_rkl"
  "mimo-jsd|scripts/eval/mimo_7b_rl/submit_four.sh|outputs/mimo_7b_rl/st_tt_jsd_clip005_1e_6_openthoughts_mimo7b/3387463/checkpoint-100|st_tt_jsd_clip005_1e_6_openthoughts_mimo7b_ckpt100|mimo_jsd"
  "mimo-rkl|scripts/eval/mimo_7b_rl/submit_four.sh|outputs/mimo_7b_rl/st_tt_rkl_clip005_1e_6_openthoughts_mimo7b/3387464/checkpoint-100|st_tt_rkl_clip005_1e_6_openthoughts_mimo7b_ckpt100|mimo_rkl"
)

echo "[jsd-rkl-seed] root=${ROOT} seeds=${SEEDS[*]} runs=${#JOBS[@]}"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r key submit_four ckpt tag prefix <<<"${entry}"
  ckpt_abs="${ROOT}/${ckpt}"
  if [[ ! -f "${ckpt_abs}/config.json" ]]; then
    echo "[error] missing checkpoint: ${ckpt_abs}" >&2
    exit 1
  fi
  for seed in "${SEEDS[@]}"; do
    echo "[jsd-rkl-seed] ${key} seed=${seed} tag=${tag}_seed${seed}"
    CHECKPOINT_PATH="${ckpt_abs}" \
      EVAL_TAG="${tag}_seed${seed}" \
      JOB_PREFIX="${prefix}_s${seed}" \
      SEED="${seed}" \
      BASE_DIR="${ROOT}" \
      bash "${ROOT}/${submit_four}"
  done
done
echo "[jsd-rkl-seed] submitted $((${#JOBS[@]} * ${#SEEDS[@]})) runs × 4 datasets"
