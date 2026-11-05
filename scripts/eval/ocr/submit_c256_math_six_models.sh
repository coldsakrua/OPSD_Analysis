#!/bin/bash
# Submit OCR c256 math evals for the same 6 models as c1024:
#   1.7b / 4b / 8b / 4b_instruct / olmo7bt / olmo7bi
# × aime24 / aime25 / aime26 / hmmt25  → 24 jobs.
#
# EVAL_TAG uses ocr_c256_* so outputs do not overwrite ocr_*_ckpt100 (c1024).
# Job names: ocr_{1p7b,4b,8b,4bi,olmo7bt,olmo7bi}_c256_{a24,a25,a26,h25}
#
# Usage (from OPSD_Analysis):
#   bash scripts/eval/ocr/submit_c256_math_six_models.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

SEED=${SEED:-42}

declare -a JOBS=(
  # model_key | submit_four | ckpt | eval_tag | job_prefix
  "1.7b|scripts/eval/1.7b/submit_four.sh|outputs/qwen3_1.7b/st_tt_clip005_1e_6_opencodereasoning_1p7b_c256/3940537/checkpoint-100|ocr_c256_qwen3_1.7b_ckpt100|ocr_1p7b_c256"
  "4b|scripts/eval/4b/submit_four.sh|outputs/qwen3_4b/st_tt_clip005_1e_6_opencodereasoning_4b_c256/3940538/checkpoint-100|ocr_c256_qwen3_4b_ckpt100|ocr_4b_c256"
  "8b|scripts/eval/8b/submit_four.sh|outputs/qwen3_8b/st_tt_clip005_1e_6_opencodereasoning_8b_c256/3949303/checkpoint-100|ocr_c256_qwen3_8b_ckpt100|ocr_8b_c256"
  "4b_instruct|scripts/eval/4b_instruct/submit_four.sh|outputs/qwen3_4b_instruct/snt_tnt_clip005_1e_6_opencodereasoning_instruct_c256/3940540/checkpoint-100|ocr_c256_qwen3_4b_instruct_ckpt100|ocr_4bi_c256"
  "olmo_7b_think|scripts/eval/olmo_7b_think/submit_four.sh|outputs/olmo3_7b_think/st_tt_clip005_1e_6_opencodereasoning_olmo7bt_c256/3949304/checkpoint-100|ocr_c256_olmo3_7b_think_ckpt100|ocr_olmo7bt_c256"
  "olmo3_7b_instruct|scripts/eval/olmo3_7b_instruct/submit_four.sh|outputs/olmo3_7b_instruct/snt_tnt_clip005_1e_6_opencodereasoning_olmo7bit_c256/3967550/checkpoint-100|ocr_c256_olmo3_7b_instruct_ckpt100|ocr_olmo7bi_c256"
)

ckpt_ok() {
  local p=$1
  [[ -f "${p}/config.json" ]] || return 1
  [[ -f "${p}/model.safetensors" ]] \
    || [[ -f "${p}/model.safetensors.index.json" ]] \
    || [[ -f "${p}/model-00001-of-00001.safetensors" ]] \
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

echo "[ocr-c256-math] root=${ROOT} seed=${SEED}"
n_models=0
for entry in "${JOBS[@]}"; do
  IFS='|' read -r _key submit_four ckpt eval_tag job_prefix <<<"${entry}"
  ckpt_abs="${ROOT}/${ckpt}"
  if ! ckpt_ok "${ckpt_abs}"; then
    echo "[error] incomplete checkpoint: ${ckpt_abs}" >&2
    exit 1
  fi
  if [[ ! -f "${ROOT}/${submit_four}" ]]; then
    echo "[error] missing ${submit_four}" >&2
    exit 1
  fi
  echo "[ocr-c256-math] ${_key} -> ${eval_tag} prefix=${job_prefix}"
  echo "  ckpt=${ckpt_abs}"
  CHECKPOINT_PATH="${ckpt_abs}" \
    EVAL_TAG="${eval_tag}" \
    JOB_PREFIX="${job_prefix}" \
    SEED="${SEED}" \
    BASE_DIR="${ROOT}" \
    bash "${ROOT}/${submit_four}"
  n_models=$((n_models + 1))
done

echo "[ocr-c256-math] submitted ${n_models} models × 4 datasets = $((n_models * 4)) jobs"
squeue -u "${USER}" -o '%.18i %.9P %.30j %.2t %.10M %R' | grep -E 'ocr_.*_c256|JOBID' || true
