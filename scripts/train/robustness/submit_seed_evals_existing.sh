#!/bin/bash
# Part A: multi-seed eval (1024/65536) on existing seed=42 variant checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/robustness"
mkdir -p "${OUT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="${OUT_DIR}/seed_eval_existing_${STAMP}.tsv"
MISS="${OUT_DIR}/seed_eval_missing_${STAMP}.txt"
: >"${REPORT}"
: >"${MISS}"
echo -e "# model\tvariant\teval_seed\tcheckpoint\tstatus" >>"${REPORT}"

ckpt_ok() {
  local p=$1
  [[ -f "${p}/config.json" ]] || return 1
  [[ -f "${p}/model.safetensors" ]] \
    || [[ -f "${p}/model.safetensors.index.json" ]] \
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

find_latest_ckpt100() {
  local root=$1
  [[ -d "${root}" ]] || return 1
  local d
  for d in $(ls -1d "${root}"/*/ 2>/dev/null | sort -V -r); do
    if ckpt_ok "${d}checkpoint-100"; then
      echo "${d}checkpoint-100"
      return 0
    fi
  done
  return 1
}

declare -A RUNS_1P7=(
  [c256]=st_tt_clip005_1e_6_ot_1p7b_c256
  [c1024]=st_tt_clip005_1e_6_openthoughts_1p7b
  [answer]=st_tt_clip005_1e_6_openthoughts_answer_1p7b
  [ios]=st_tt_clip005_1e_6_openthoughts_irr_other_sol_1p7b
  [first256]=st_tt_clip005_c1024_first256_1p7b
  [uni256]=st_tt_clip005_c1024_uni256_1p7b
  [last256]=st_tt_clip005_c1024_last256_1p7b
)

declare -A RUNS_OLMO=(
  [c256]=st_tt_clip005_1e_6_ot_olmo7bt_c256
  [c1024]=st_tt_clip005_1e_6_openthoughts_olmo7bt
  [answer]=st_tt_clip005_1e_6_openthoughts_answer_olmo7bt
  [ios]=st_tt_clip005_1e_6_openthoughts_irr_other_sol_olmo7bt
  [first256]=st_tt_clip005_c1024_first256_olmo7bt
  [uni256]=st_tt_clip005_c1024_uni256_olmo7bt
  [last256]=st_tt_clip005_c1024_last256_olmo7bt
)

VARIANTS=(c256 c1024 answer ios first256 uni256 last256)
EVAL_SEEDS=(1024 65536)

abs_ckpt() {
  local p=$1
  if [[ "${p}" = /* ]]; then echo "${p}"; else echo "${ROOT}/${p}"; fi
}

echo "[seed-eval] submitting multi-seed evals on existing ckpts"

for v in "${VARIANTS[@]}"; do
  run="${RUNS_1P7[$v]}"
  if ckpt=$(find_latest_ckpt100 "${ROOT}/outputs/qwen3_1.7b/${run}"); then
    ckpt=$(abs_ckpt "${ckpt}")
    for es in "${EVAL_SEEDS[@]}"; do
      echo "[seed-eval] 1.7b ${v} seed=${es} -> ${ckpt}"
      if CHECKPOINT_PATH="${ckpt}" BASE_DIR="${ROOT}" \
          bash "${ROOT}/scripts/eval/1.7b/seed/submit_four_seed${es}.sh"; then
        echo -e "qwen3_1.7b\t${v}\t${es}\t${ckpt}\tOK" >>"${REPORT}"
      else
        echo -e "qwen3_1.7b\t${v}\t${es}\t${ckpt}\tFAIL" >>"${REPORT}"
        echo "FAIL submit 1.7b ${v} seed=${es}" >>"${MISS}"
      fi
    done
  else
    echo "[MISSING] 1.7b ${v} (run=${run})" | tee -a "${MISS}"
    echo -e "qwen3_1.7b\t${v}\t-\tMISSING\tMISSING" >>"${REPORT}"
  fi
done

for v in "${VARIANTS[@]}"; do
  run="${RUNS_OLMO[$v]}"
  if ckpt=$(find_latest_ckpt100 "${ROOT}/outputs/olmo3_7b_think/${run}"); then
    ckpt=$(abs_ckpt "${ckpt}")
    for es in "${EVAL_SEEDS[@]}"; do
      echo "[seed-eval] olmo ${v} seed=${es} -> ${ckpt}"
      if CHECKPOINT_PATH="${ckpt}" BASE_DIR="${ROOT}" \
          bash "${ROOT}/scripts/eval/olmo_7b_think/seed/submit_four_seed${es}.sh"; then
        echo -e "olmo3_7b_think\t${v}\t${es}\t${ckpt}\tOK" >>"${REPORT}"
      else
        echo -e "olmo3_7b_think\t${v}\t${es}\t${ckpt}\tFAIL" >>"${REPORT}"
        echo "FAIL submit olmo ${v} seed=${es}" >>"${MISS}"
      fi
    done
  else
    echo "[MISSING] olmo ${v} (run=${run})" | tee -a "${MISS}"
    echo -e "olmo3_7b_think\t${v}\t-\tMISSING\tMISSING" >>"${REPORT}"
  fi
done

echo "[seed-eval] report=${REPORT}"
echo "[seed-eval] missing=${MISS}"
cat "${REPORT}"
if [[ -s "${MISS}" ]]; then
  echo "----- missing/fail -----"
  cat "${MISS}"
fi
