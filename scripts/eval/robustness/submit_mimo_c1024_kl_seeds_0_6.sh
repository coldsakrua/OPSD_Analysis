#!/bin/bash
# Eval seeds 0-6 for MiMo-7B-RL base and the three c1024 KL checkpoints.
# Existing seeds 42 / 1024 / 65536 are left in place. Does not retrain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

# key | checkpoint | base eval tag | job prefix
JOBS=(
  "base|/dfs/share/maleiLab/2501210611/model/mimo-7b-rl|mimo-7b-rl|mimo_base"
  "fwd|outputs/mimo_7b_rl/st_tt_clip005_1e_6_openthoughts_mimo7b/3289222/checkpoint-100|st_tt_clip005_1e6_mimo7b_ckpt100|mimo_fwd"
  "jsd|outputs/mimo_7b_rl/st_tt_jsd_clip005_1e_6_openthoughts_mimo7b/3387463/checkpoint-100|st_tt_jsd_clip005_1e_6_openthoughts_mimo7b_ckpt100|mimo_jsd"
  "rkl|outputs/mimo_7b_rl/st_tt_rkl_clip005_1e_6_openthoughts_mimo7b/3387464/checkpoint-100|st_tt_rkl_clip005_1e_6_openthoughts_mimo7b_ckpt100|mimo_rkl"
)

for entry in "${JOBS[@]}"; do
  IFS='|' read -r key ckpt tag prefix <<<"${entry}"
  if [[ "${ckpt}" = /* ]]; then
    ckpt_abs="${ckpt}"
  else
    ckpt_abs="${ROOT}/${ckpt}"
  fi
  if [[ ! -f "${ckpt_abs}/config.json" ]]; then
    echo "[error] missing checkpoint: ${ckpt_abs}" >&2
    exit 1
  fi
  for seed in 0 1 2 3 4 5 6; do
    echo "[mimo-seed] ${key} seed=${seed} tag=${tag}_seed${seed}"
    CHECKPOINT_PATH="${ckpt_abs}" \
      EVAL_TAG="${tag}_seed${seed}" \
      JOB_PREFIX="${prefix}_s${seed}" \
      SEED="${seed}" \
      BASE_DIR="${ROOT}" \
      bash "${ROOT}/scripts/eval/mimo_7b_rl/submit_four.sh"
  done
done
echo "[mimo-seed] submitted base and three c1024 KL runs, seeds 0-6"
