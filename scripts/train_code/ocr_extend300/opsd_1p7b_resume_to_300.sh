#!/bin/bash
#SBATCH --job-name=st_tt_ocr_1p7b_r300
#SBATCH --output=log/train_code/1.7b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=8:00:00
set -euo pipefail

# Resume Qwen3-1.7B OCR OPSD (job 3908495) from checkpoint-100 → max_steps=300.
# Same OUTPUT_DIR so HF Trainer continues dataset skip from global_step=100.
# Saves every 25 steps; watcher submits code evals every 50 steps (150/200/250/300).
#
# NOTE: Slurm copies this script under /var/spool/... so never derive BASE_DIR
# from BASH_SOURCE. Prefer explicit BASE_DIR / SLURM_SUBMIT_DIR / hardcoded root.

export BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
if [[ ! -d "${BASE_DIR}/outputs" || ! -d "${BASE_DIR}/scripts" ]]; then
  BASE_DIR=/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis
  export BASE_DIR
fi

RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_opencodereasoning_1p7b}
MODEL_TAG=${MODEL_TAG:-qwen3_1.7b}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/${MODEL_TAG}}
OUTPUT_JID=${OUTPUT_JID:-3908495}

export OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/${OUTPUT_JID}}
export RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-checkpoint-100}
export MAX_STEPS=${MAX_STEPS:-300}
export SAVE_STEPS=${SAVE_STEPS:-25}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-20}
export RUN_NAME_WITH_JOB=${RUN_NAME_WITH_JOB:-${RUN_NAME}_${OUTPUT_JID}}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_1p7b_fullparam_300step_opencodereasoning}

if [[ ! -d "${OUTPUT_DIR}/${RESUME_FROM_CHECKPOINT}" && ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "[error] missing resume ckpt under ${OUTPUT_DIR}: ${RESUME_FROM_CHECKPOINT}" >&2
  exit 1
fi

echo "[resume-1p7b-ocr] BASE_DIR=${BASE_DIR}"
echo "[resume-1p7b-ocr] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[resume-1p7b-ocr] RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT} MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS}"

cd "${BASE_DIR}"
exec bash "${BASE_DIR}/scripts/train_code/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh"
