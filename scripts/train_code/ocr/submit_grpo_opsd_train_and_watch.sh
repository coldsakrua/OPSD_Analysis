#!/bin/bash
# Submit 4B GRPO→OPSD trains (gs50 / gs100 init) + CPU watcher.
# Watcher submits think-mode LCB / HumanEval+ / MBPP+ generate jobs at ckpt-100.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train_code/ocr/submit_grpo_opsd_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train_code/ocr_grpo_opsd"
mkdir -p "${OUT_DIR}" log/train_code/4b_base
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

DATASET_PATH=${DATASET_PATH:-${ROOT}/data/opencodereasoning/preprocessed/ocr.opsd.solution.shared.qwen_olmo.tmax2048.parquet}

declare -a JOBS=(
  "qwen3_4b_base|qwen3_4b_base|st_tt_clip005_1e_6_opencodereasoning_4b_grpo50|scripts/train_code/qwen3_4b_base/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning_grpo50.sh|ocr_4b_grpo50_ckpt100"
  "qwen3_4b_base|qwen3_4b_base|st_tt_clip005_1e_6_opencodereasoning_4b_grpo100|scripts/train_code/qwen3_4b_base/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning_grpo100.sh|ocr_4b_grpo100_ckpt100"
)

chmod +x \
  scripts/train_code/qwen3_4b_base/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning_grpo50.sh \
  scripts/train_code/qwen3_4b_base/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning_grpo100.sh \
  scripts/train_code/ocr/sbatch_watch.sh \
  scripts/train_code/ocr/watch_then_eval.sh \
  scripts/train_code/ocr/submit_three_code_evals.sh \
  scripts/train_code/ocr/submit_grpo_opsd_train_and_watch.sh

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing dataset ${DATASET_PATH}" >&2
  exit 1
fi
echo "[submit] dataset ready: ${DATASET_PATH}"

for ckpt in \
  "${ROOT}/outputs/qwen3_4b_base/grpo_think_4b_ocr_sft15000_apps_4gpu/4183528/global_step_50/actor/hf_merged" \
  "${ROOT}/outputs/qwen3_4b_base/grpo_think_4b_ocr_sft15000_apps_4gpu/4183528/global_step_100/actor/hf_merged"
do
  if [[ ! -f "${ckpt}/config.json" ]]; then
    echo "[error] missing GRPO hf_merged: ${ckpt}" >&2
    exit 1
  fi
  echo "[submit] grpo ckpt ok: ${ckpt}"
done

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
  --job-name=watch_ocr_grpo_opsd \
  --output=log/train_code/ocr_grpo_opsd/watch_slurm.%j.out \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}",CKPT_STEP=100 \
  "${ROOT}/scripts/train_code/ocr/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done (evals: lcb_v6 / humanevalplus / mbppplus think @ checkpoint-100)"
cat "${MANIFEST}"
