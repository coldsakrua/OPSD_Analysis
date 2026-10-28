#!/bin/bash
# Submit Qwen3-1.7B / 4B / 8B APPS OPSD trains + CPU watcher.
# Trains are 50-step; watcher submits think-mode LCB / HumanEval+ / MBPP+ at ckpt-50.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train_code/ocr/submit_apps_opsd_train_and_watch.sh
#   MODELS=8b bash scripts/train_code/ocr/submit_apps_opsd_train_and_watch.sh   # only 8B
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train_code/apps_opsd"
mkdir -p "${OUT_DIR}" log/train_code/{1.7b,4b,8b}
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"
CKPT_STEP=${CKPT_STEP:-50}
MODELS=${MODELS:-all}

DATASET_PATH=${DATASET_PATH:-${ROOT}/data/apps/preprocessed/apps.opsd.solution.sthink_tthink.maxprompt2048.parquet}

declare -a ALL_JOBS=(
  "qwen3_1.7b|qwen3_1.7b|st_tt_clip005_1e_6_apps_1p7b|scripts/train_code/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_apps.sh|apps_opsd_qwen3_1p7b_ckpt${CKPT_STEP}|/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b"
  "qwen3_4b|qwen3_4b|st_tt_clip005_1e_6_apps_4b|scripts/train_code/qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_apps.sh|apps_opsd_qwen3_4b_ckpt${CKPT_STEP}|/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b"
  "qwen3_8b|qwen3_8b|st_tt_clip005_1e_6_apps_8b|scripts/train_code/qwen3_8b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_apps.sh|apps_opsd_qwen3_8b_ckpt${CKPT_STEP}|/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-8b"
)

declare -a JOBS=()
for entry in "${ALL_JOBS[@]}"; do
  IFS='|' read -r model_key _rest <<<"${entry}"
  case "${MODELS}" in
    all) JOBS+=("${entry}") ;;
    1.7b|1p7b)
      [[ "${model_key}" == "qwen3_1.7b" ]] && JOBS+=("${entry}")
      ;;
    4b)
      [[ "${model_key}" == "qwen3_4b" ]] && JOBS+=("${entry}")
      ;;
    8b)
      [[ "${model_key}" == "qwen3_8b" ]] && JOBS+=("${entry}")
      ;;
    *)
      echo "[error] unknown MODELS=${MODELS} (use all|1.7b|4b|8b)" >&2
      exit 1
      ;;
  esac
done

if [[ "${#JOBS[@]}" -eq 0 ]]; then
  echo "[error] no jobs selected (MODELS=${MODELS})" >&2
  exit 1
fi

chmod +x \
  scripts/train_code/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_apps.sh \
  scripts/train_code/qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_apps.sh \
  scripts/train_code/qwen3_8b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_apps.sh \
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

for entry in "${JOBS[@]}"; do
  IFS='|' read -r _model_key _model_tag _run_name _script _eval_tag model_path <<<"${entry}"
  if [[ ! -d "${model_path}" ]]; then
    echo "[error] missing model: ${model_path}" >&2
    exit 1
  fi
  echo "[submit] model ok: ${model_path}"
done

{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript\teval_tag"
} >"${MANIFEST_STAMPED}"

echo "[submit] root=${ROOT} MODELS=${MODELS} CKPT_STEP=${CKPT_STEP}"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r model_key model_tag run_name script eval_tag _model_path <<<"${entry}"
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
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}",CKPT_STEP="${CKPT_STEP}" \
  "${ROOT}/scripts/train_code/ocr/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done (evals: lcb_v6 / humanevalplus / mbppplus think @ checkpoint-${CKPT_STEP})"
cat "${MANIFEST}"
