#!/bin/bash
# Submit cross-model OCR trains (priv/opsd + same) + CPU watcher for code evals:
#   1) student=qwen3-1.7b, teacher=qwen3-4b (think)
#   2) student=qwen3-4b, teacher=qwen3-4b-thinking
#
# Usage (from OPSD_Analysis):
#   bash scripts/train_code/ocr/submit_cross_xd_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train_code/ocr_cross_xd"
mkdir -p "${OUT_DIR}" \
  log/train_code/s1.7b_t4b \
  log/train_code/s4b_t4bt \
  log/eval_code/1.7b/{lcb_v6,humanevalplus,mbppplus}/think \
  log/eval_code/4b/{lcb_v6,humanevalplus,mbppplus}/think

MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

DATASET_PATH=${DATASET_PATH:-${ROOT}/data/opencodereasoning/preprocessed/ocr.opsd.solution.shared.qwen_olmo.tmax2048.parquet}

declare -a JOBS=(
  "qwen3_1.7b|s1.7b_t4b|s1p7_t4b_st_tt_clip005_1e_6_opencodereasoning|scripts/train_code/s1.7b_t4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh|ocr_s1p7_t4b_opsd_ckpt100"
  "qwen3_1.7b|s1.7b_t4b|s1p7_t4b_st_tt_same_clip005_1e_6_opencodereasoning|scripts/train_code/s1.7b_t4b/jsd005/same_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh|ocr_s1p7_t4b_same_ckpt100"
  "qwen3_4b|s4b_t4bt|s4b_t4bt_st_tt_clip005_1e_6_opencodereasoning|scripts/train_code/s4b_t4bt/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh|ocr_s4b_t4bt_opsd_ckpt100"
  "qwen3_4b|s4b_t4bt|s4b_t4bt_st_tt_same_clip005_1e_6_opencodereasoning|scripts/train_code/s4b_t4bt/jsd005/same_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh|ocr_s4b_t4bt_same_ckpt100"
)

chmod +x \
  scripts/train_code/s1.7b_t4b/jsd005/*.sh \
  scripts/train_code/s4b_t4bt/jsd005/*.sh \
  scripts/train_code/ocr/sbatch_watch.sh \
  scripts/train_code/ocr/watch_then_eval.sh \
  scripts/train_code/ocr/submit_three_code_evals.sh \
  scripts/eval_code/1.7b/*.sh \
  scripts/eval_code/4b/*.sh

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing dataset ${DATASET_PATH}" >&2
  exit 1
fi
echo "[submit] dataset ready: ${DATASET_PATH}"

{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript\teval_tag"
} >"${MANIFEST_STAMPED}"

echo "[submit] root=${ROOT}"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r model_key model_tag run_name script eval_tag <<<"${entry}"
  if [[ ! -f "${script}" ]]; then
    echo "[error] missing ${script}" >&2
    exit 1
  fi
  jid=$(sbatch --parsable --chdir="${ROOT}" "${script}")
  jid="${jid%%;*}"
  echo -e "${model_key}\t${model_tag}\t${run_name}\t${jid}\t${script}\t${eval_tag}" >>"${MANIFEST_STAMPED}"
  echo "[submit] ${eval_tag} -> train_jid=${jid} script=${script}"
done

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"
echo "[submit] stamped=${MANIFEST_STAMPED}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_ocr_cross_xd \
  --output=log/train_code/ocr_cross_xd/watch_slurm.%j.out \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}",CKPT_STEP=100 \
  "${ROOT}/scripts/train_code/ocr/sbatch_watch.sh")
watch_jid="${watch_jid%%;*}"
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
