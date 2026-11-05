#!/bin/bash
# Submit OpenThoughts c256 coding evals for the same 6 models as c1024 OT code:
#   1.7b / 4b / 8b / 4b_instruct / olmo7bt / olmo7bi
# × LCB / HumanEval+ / MBPP+  → 18 generate jobs (judges auto-submit).
#
# Mirrors c1024 tags like ot_st_tt_qwen3_1p7b_ckpt100, with _c256_ so outputs
# do not overwrite c1024 eval_outputs/ot_*_ckpt100.
#
# Usage (from OPSD_Analysis):
#   bash scripts/eval_code/ot/submit_c256_code_six_models.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

SEED=${SEED:-42}
SUBMIT="${ROOT}/scripts/train_code/ocr/submit_three_code_evals.sh"
chmod +x "${SUBMIT}"

declare -a JOBS=(
  # model_key | ckpt | eval_tag
  "qwen3_1.7b|outputs/qwen3_1.7b/st_tt_clip005_1e_6_ot_1p7b_c256/3226382/checkpoint-100|ot_st_tt_qwen3_1p7b_c256_ckpt100"
  "qwen3_4b|outputs/qwen3_4b/st_tt_clip005_1e_6_ot_4b_c256/3226380/checkpoint-100|ot_st_tt_qwen3_4b_c256_ckpt100"
  "qwen3_8b|outputs/qwen3_8b/st_tt_clip005_1e_6_ot_8b_c256/3346472/checkpoint-100|ot_st_tt_qwen3_8b_c256_ckpt100"
  "qwen3_4b_instruct|outputs/qwen3_4b_instruct/snt_tnt_clip005_1e_6_oti_c256/3226358/checkpoint-100|ot_snt_tnt_qwen3_4b_instruct_c256_ckpt100"
  "olmo3_7b_think|outputs/olmo3_7b_think/st_tt_clip005_1e_6_ot_olmo7bt_c256/3236843/checkpoint-100|ot_st_tt_olmo3_7b_think_c256_ckpt100"
  "olmo3_7b_instruct|outputs/olmo3_7b_instruct/snt_tnt_clip005_1e_6_oti_olmo7bi_c256/3226917/checkpoint-100|ot_snt_tnt_olmo3_7b_it_c256_ckpt100"
)

ckpt_ok() {
  local p=$1
  [[ -f "${p}/config.json" ]] || return 1
  [[ -f "${p}/model.safetensors" ]] \
    || [[ -f "${p}/model.safetensors.index.json" ]] \
    || [[ -f "${p}/model-00001-of-00001.safetensors" ]] \
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

echo "[ot-c256-code] root=${ROOT} seed=${SEED}"
n=0
for entry in "${JOBS[@]}"; do
  IFS='|' read -r model_key ckpt eval_tag <<<"${entry}"
  ckpt_abs="${ROOT}/${ckpt}"
  if ! ckpt_ok "${ckpt_abs}"; then
    echo "[error] incomplete checkpoint: ${ckpt_abs}" >&2
    exit 1
  fi
  echo "[ot-c256-code] ${model_key} -> ${eval_tag}"
  echo "  ckpt=${ckpt_abs}"
  MODEL_KEY="${model_key}" \
    CHECKPOINT_PATH="${ckpt_abs}" \
    EVAL_TAG="${eval_tag}" \
    SEED="${SEED}" \
    AUTO_JUDGE=1 \
    BASE_DIR="${ROOT}" \
    bash "${SUBMIT}"
  n=$((n + 1))
done

echo "[ot-c256-code] submitted ${n} models × 3 datasets = $((n * 3)) generate jobs"
squeue -u "${USER}" -o '%.18i %.9P %.40j %.2t %.10M %R' | grep -E 'ot_.*_c256|JOBID' || true
