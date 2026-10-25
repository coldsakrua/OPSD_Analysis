#!/bin/bash
# Submit 1.7B GRPO@100 → OPSD train + CPU watcher.
# Watcher submits think-mode LCB / HumanEval+ / MBPP+ generate jobs at ckpt-100.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train_code/ocr/submit_1p7b_grpo100_opsd_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train_code/ocr_grpo_opsd"
mkdir -p "${OUT_DIR}" log/train_code/1.7b_base
MANIFEST="${OUT_DIR}/submit_1p7b_grpo100_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_1p7b_grpo100_${STAMP}.tsv"

DATASET_PATH=${DATASET_PATH:-${ROOT}/data/opencodereasoning/preprocessed/ocr.opsd.solution.shared.qwen_olmo.tmax2048.parquet}
GRPO_CKPT=${GRPO_CKPT:-${ROOT}/outputs/qwen3_1.7b_base/grpo_think_1p7b_ocr_sft15000_apps_4gpu/4245276/global_step_100/actor/hf_merged}
TRAIN_SCRIPT=scripts/train_code/qwen3_1.7b_base/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning_grpo100.sh

chmod +x \
  "${TRAIN_SCRIPT}" \
  scripts/train_code/ocr/sbatch_watch.sh \
  scripts/train_code/ocr/watch_then_eval.sh \
  scripts/train_code/ocr/submit_three_code_evals.sh \
  scripts/train_code/ocr/submit_1p7b_grpo100_opsd_train_and_watch.sh

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing dataset ${DATASET_PATH}" >&2
  exit 1
fi
echo "[submit] dataset ready: ${DATASET_PATH}"

if [[ ! -f "${GRPO_CKPT}/config.json" ]]; then
  echo "[error] missing GRPO hf_merged: ${GRPO_CKPT}" >&2
  echo "[error] merge FSDP first, e.g. sbatch scripts/train_code/sft_rl/merge_gs100_1p7b_apps_4245276.sh" >&2
  exit 1
fi
echo "[submit] grpo ckpt ok: ${GRPO_CKPT}"

{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript\teval_tag"
} >"${MANIFEST_STAMPED}"

echo "[submit] root=${ROOT}"
jid=$(sbatch --parsable --chdir="${ROOT}" "${TRAIN_SCRIPT}")
echo -e "qwen3_1.7b_base\tqwen3_1.7b_base\tst_tt_clip005_1e_6_opencodereasoning_1p7b_grpo100\t${jid}\t${TRAIN_SCRIPT}\tocr_1p7b_grpo100_ckpt100" >>"${MANIFEST_STAMPED}"
echo "[submit] ocr_1p7b_grpo100_ckpt100 -> train_jid=${jid} script=${TRAIN_SCRIPT}"

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_ocr_1p7b_g100 \
  --output=log/train_code/ocr_grpo_opsd/watch_1p7b_grpo100_slurm.%j.out \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}",CKPT_STEP=100 \
  "${ROOT}/scripts/train_code/ocr/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done (evals: lcb_v6 / humanevalplus / mbppplus think @ checkpoint-100)"
cat "${MANIFEST}"
