#!/bin/bash
# Submit Qwen3-1.7B / 4B APPS OPSD trains + CPU watcher.
# Watcher submits think-mode LCB / HumanEval+ / MBPP+ generate jobs at ckpt-100.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train_code/ocr/submit_apps_opsd_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train_code/apps_opsd"
mkdir -p "${OUT_DIR}" log/train_code/{1.7b,4b}
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

DATASET_PATH=${DATASET_PATH:-${ROOT}/data/apps/preprocessed/apps.opsd.solution.sthink_tthink.maxprompt2048.parquet}

declare -a JOBS=(
  "qwen3_1.7b|qwen3_1.7b|st_tt_clip005_1e_6_apps_1p7b|scripts/train_code/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_apps.sh|apps_opsd_qwen3_1p7b_ckpt100"
  "qwen3_4b|qwen3_4b|st_tt_clip005_1e_6_apps_4b|scripts/train_code/qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_apps.sh|apps_opsd_qwen3_4b_ckpt100"
)

chmod +x \
  scripts/train_code/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_apps.sh \
  scripts/train_code/qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_apps.sh \
  scripts/train_code/ocr/sbatch_watch.sh \
  scripts/train_code/ocr/watch_then_eval.sh \
  scripts/train_code/ocr/submit_three_code_evals.sh \
  scripts/train_code/ocr/submit_apps_opsd_train_and_watch.sh

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing APPS OPSD dataset ${DATASET_PATH}" >&2
  echo "[error] run: python scripts/data/preprocess_opsd_apps.py --min-rows 6400" >&2
  exit 1
fi
echo "[submit] dataset ready: ${DATASET_PATH}"

for model in \
  /gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b \
  /gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b
do
  if [[ ! -d "${model}" ]]; then
    echo "[error] missing model: ${model}" >&2
    exit 1
  fi
  echo "[submit] model ok: ${model}"
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
  --job-name=watch_apps_opsd \
  --output=log/train_code/apps_opsd/watch_slurm.%j.out \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}",CKPT_STEP=100 \
  "${ROOT}/scripts/train_code/ocr/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done (evals: lcb_v6 / humanevalplus / mbppplus think @ checkpoint-100)"
cat "${MANIFEST}"
