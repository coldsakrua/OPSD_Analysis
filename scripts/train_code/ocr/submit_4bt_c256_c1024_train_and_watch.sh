#!/bin/bash
# Submit Qwen3-4B-Thinking OCR OPSD trains (c256 + c1024) + CPU watcher,
# and optionally base-model code evals (LCB / HumanEval+ / MBPP+).
#
# Usage (from OPSD_Analysis):
#   bash scripts/train_code/ocr/submit_4bt_c256_c1024_train_and_watch.sh
# Skip base evals:
#   SUBMIT_BASE_EVAL=0 bash scripts/train_code/ocr/submit_4bt_c256_c1024_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train_code/ocr_4bt"
mkdir -p "${OUT_DIR}" log/train_code/4b_thinking
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

DATASET_PATH=${DATASET_PATH:-${ROOT}/data/opencodereasoning/preprocessed/ocr.opsd.solution.shared.qwen_olmo.tmax2048.parquet}
SUBMIT_BASE_EVAL=${SUBMIT_BASE_EVAL:-1}
BASE_MODEL=${BASE_MODEL:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b-thinking}
BASE_EVAL_TAG=${BASE_EVAL_TAG:-qwen3-4b-thinking}

declare -a JOBS=(
  "qwen3_4b_thinking|qwen3_4b_thinking|st_tt_clip005_1e_6_opencodereasoning_4bt_c256|scripts/train_code/qwen3_4b_thinking/jsd005/length/opsd_st_tt_clip005_c256_1e6_ocr.sh|ocr_c256_qwen3_4b_thinking_ckpt100"
  "qwen3_4b_thinking|qwen3_4b_thinking|st_tt_clip005_1e_6_opencodereasoning_4bt_c1024|scripts/train_code/qwen3_4b_thinking/jsd005/length/opsd_st_tt_clip005_c1024_1e6_ocr.sh|ocr_c1024_qwen3_4b_thinking_ckpt100"
)

chmod +x \
  scripts/train_code/qwen3_4b_thinking/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh \
  scripts/train_code/qwen3_4b_thinking/jsd005/length/opsd_st_tt_clip005_c256_1e6_ocr.sh \
  scripts/train_code/qwen3_4b_thinking/jsd005/length/opsd_st_tt_clip005_c1024_1e6_ocr.sh \
  scripts/train_code/ocr/sbatch_watch.sh \
  scripts/train_code/ocr/watch_then_eval.sh \
  scripts/train_code/ocr/submit_three_code_evals.sh \
  scripts/eval_code/4b_thinking/*.sh

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
  echo -e "${model_key}\t${model_tag}\t${run_name}\t${jid}\t${script}\t${eval_tag}" >>"${MANIFEST_STAMPED}"
  echo "[submit] ${eval_tag} -> train_jid=${jid} script=${script}"
done

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_ocr_4bt \
  --output=log/train_code/ocr_4bt/watch_slurm.%j.out \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}",CKPT_STEP=100 \
  "${ROOT}/scripts/train_code/ocr/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"

if [[ "${SUBMIT_BASE_EVAL}" == "1" ]]; then
  echo "[submit] base evals MODEL_KEY=qwen3_4b_thinking ckpt=${BASE_MODEL} tag=${BASE_EVAL_TAG}"
  MODEL_KEY=qwen3_4b_thinking \
  CHECKPOINT_PATH="${BASE_MODEL}" \
  EVAL_TAG="${BASE_EVAL_TAG}" \
  BASE_DIR="${ROOT}" \
  bash "${ROOT}/scripts/train_code/ocr/submit_three_code_evals.sh"
fi

echo "[submit] done"
cat "${MANIFEST}"
