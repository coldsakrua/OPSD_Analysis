#!/bin/bash
#SBATCH --job-name=st_tt_olmo_r300
#SBATCH --output=log/train/olmo3-7b-think/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=8:00:00
#SBATCH --exclude=gpua800n03,gpua800n10,gpua800n13,gpua800n21
set -euo pipefail

# Resume Olmo-3-7B-Think st/tt clip005 1e-6 from checkpoint-100 → max_steps=300.
# Saves every 50 steps (150/200/250/300) for think evals.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BASE_DIR=${BASE_DIR:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}
RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_openthoughts_olmo7bt}
MODEL_TAG=${MODEL_TAG:-olmo3_7b_think}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/${MODEL_TAG}}

export OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/3233311}
export RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-checkpoint-100}
export MAX_STEPS=${MAX_STEPS:-300}
export SAVE_STEPS=${SAVE_STEPS:-50}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-12}
export RUN_NAME_WITH_JOB=${RUN_NAME_WITH_JOB:-${RUN_NAME}_3233311}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-olmo3_7b_think_fullparam_300step_openthoughts}

if [[ ! -d "${OUTPUT_DIR}/${RESUME_FROM_CHECKPOINT}" && ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "[error] missing resume ckpt under ${OUTPUT_DIR}: ${RESUME_FROM_CHECKPOINT}" >&2
  exit 1
fi

echo "[resume-olmo] BASE_DIR=${BASE_DIR}"
echo "[resume-olmo] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[resume-olmo] RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT} MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS}"

cd "${BASE_DIR}"
exec bash "${BASE_DIR}/scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts.sh"

