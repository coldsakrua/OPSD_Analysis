#!/bin/bash
# Submit GSM8K OPSD st_tt trains (1.7b / 4b / olmo7bt) + CPU watcher for ckpt-100 think evals.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/gsm8k_st_tt/submit_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/gsm8k_st_tt"
mkdir -p "${OUT_DIR}" log/train/{1.7b,4b,olmo3-7b-think}
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

DATASET_PATH=${DATASET_PATH:-${ROOT}/data/gsm8k/preprocessed/gsm8k.opsd.solution.sthink_tthink.shared.qwen_olmo.tmax1024.parquet}
DATA_JID=${DATA_JID:-}

declare -a JOBS=(
  "qwen3_1.7b|qwen3_1.7b|st_tt_clip005_1e_6_gsm8k_1p7b|scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_gsm8k.sh|g8k_st_tt_1p7b_ckpt100"
  "qwen3_4b|qwen3_4b|st_tt_clip005_1e_6_gsm8k_4b|scripts/train/qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_gsm8k.sh|g8k_st_tt_4b_ckpt100"
  "olmo3_7b_think|olmo3_7b_think|st_tt_clip005_1e_6_gsm8k_olmo7bt|scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_gsm8k.sh|g8k_st_tt_olmo7bt_ckpt100"
)

chmod +x \
  scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_gsm8k.sh \
  scripts/train/qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_gsm8k.sh \
  scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_gsm8k.sh \
  scripts/train/gsm8k_st_tt/sbatch_watch.sh \
  scripts/train/gsm8k_st_tt/watch_then_eval.sh \
  scripts/train/gsm8k_st_tt/submit_four_think_evals.sh

dep_args=()
if [[ -f "${DATASET_PATH}" ]]; then
  echo "[submit] dataset ready: ${DATASET_PATH}"
elif [[ -n "${DATA_JID}" ]]; then
  echo "[submit] dataset missing; trains depend on DATA_JID=${DATA_JID}"
  dep_args=(--dependency="afterok:${DATA_JID}")
else
  echo "[error] missing dataset ${DATASET_PATH} and DATA_JID unset" >&2
  exit 1
fi

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
  if [[ ${#dep_args[@]} -gt 0 ]]; then
    jid=$(sbatch --parsable --chdir="${ROOT}" "${dep_args[@]}" "${script}")
  else
    jid=$(sbatch --parsable --chdir="${ROOT}" "${script}")
  fi
  echo -e "${model_key}\t${model_tag}\t${run_name}\t${jid}\t${script}\t${eval_tag}" >>"${MANIFEST_STAMPED}"
  echo "[submit] ${eval_tag} -> train_jid=${jid} script=${script}"
done

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_gsm8k_st_tt \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}",CKPT_STEP=100 \
  "${ROOT}/scripts/train/gsm8k_st_tt/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
