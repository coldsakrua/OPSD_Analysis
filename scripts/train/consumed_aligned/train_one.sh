#!/bin/bash
#SBATCH --job-name=ot_aligned
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
set -euo pipefail

# One aligned answer-only or unrelated run.
# Resources, SPEC, and EXCLUDE come from the submit command.
# Dataset row 0..K-1 is the full-solution consumption order. SequentialSampler
# plus the script's own global batch and 100 steps reads exactly those rows.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
SPEC=${SPEC:?Set SPEC}
cd "${BASE_DIR}"

DATA_ROOT="${BASE_DIR}/data/openthoughts/preprocessed/consumed_aligned"

case "${SPEC}" in
  qwen17_answer)
    INNER="scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_answer.sh"
    DATASET_PATH="${DATA_ROOT}/qwen_st_tt.answer.n6400.parquet"
    RUN_NAME="st_tt_clip005_1e_6_ot_ans_same6400_1p7b"
    ;;
  qwen17_unrelated)
    INNER="scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_irrelevant_other_sol.sh"
    DATASET_PATH="${DATA_ROOT}/qwen_st_tt.unrelated.n6400.parquet"
    RUN_NAME="st_tt_clip005_1e_6_ot_ios_same6400_1p7b"
    ;;
  qwen4b_answer)
    INNER="scripts/train/qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_answer.sh"
    DATASET_PATH="${DATA_ROOT}/qwen_st_tt.answer.n6400.parquet"
    RUN_NAME="st_tt_clip005_1e_6_ot_ans_same3200_4b"
    ;;
  qwen4b_unrelated)
    INNER="scripts/train/qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_irrelevant_other_sol.sh"
    DATASET_PATH="${DATA_ROOT}/qwen_st_tt.unrelated.n6400.parquet"
    RUN_NAME="st_tt_clip005_1e_6_ot_ios_same3200_4b"
    ;;
  qwen4bi_answer)
    INNER="scripts/train/qwen3_4b_instruct/jsd005/opsd_student_nothink_teacher_nothink_clip005_1e_6_openthoughts_answer.sh"
    DATASET_PATH="${DATA_ROOT}/qwen_st_tt.answer.n6400.parquet"
    RUN_NAME="snt_tnt_clip005_1e_6_ot_ans_same3200_instruct"
    ;;
  qwen4bi_unrelated)
    INNER="scripts/train/qwen3_4b_instruct/jsd005/opsd_student_nothink_teacher_nothink_clip005_1e_6_openthoughts_irrelevant_other_sol.sh"
    DATASET_PATH="${DATA_ROOT}/qwen_st_tt.unrelated.n6400.parquet"
    RUN_NAME="snt_tnt_clip005_1e_6_ot_ios_same3200_instruct"
    ;;
  olmo_think_answer)
    INNER="scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_answer.sh"
    DATASET_PATH="${DATA_ROOT}/olmo_think.answer.n6400.parquet"
    RUN_NAME="st_tt_clip005_1e_6_ot_ans_same6400_olmo7bt"
    ;;
  olmo_think_unrelated)
    INNER="scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_irrelevant_other_sol.sh"
    DATASET_PATH="${DATA_ROOT}/olmo_think.unrelated.n6400.parquet"
    RUN_NAME="st_tt_clip005_1e_6_ot_ios_same6400_olmo7bt"
    ;;
  olmo_it_answer)
    INNER="scripts/train/olmo3_7b_instruct/jsd005/opsd_student_nothink_teacher_nothink_clip005_1e_6_openthoughts_answer.sh"
    DATASET_PATH="${DATA_ROOT}/olmo_it.answer.n6400.parquet"
    RUN_NAME="snt_tnt_clip005_1e_6_ot_ans_same6400_olmo7bit"
    ;;
  olmo_it_unrelated)
    INNER="scripts/train/olmo3_7b_instruct/jsd005/opsd_student_nothink_teacher_nothink_clip005_1e_6_openthoughts_irrelevant_other_sol.sh"
    DATASET_PATH="${DATA_ROOT}/olmo_it.unrelated.n6400.parquet"
    RUN_NAME="snt_tnt_clip005_1e_6_ot_ios_same6400_olmo7bit"
    ;;
  *)
    echo "[error] unknown SPEC=${SPEC}" >&2
    exit 1
    ;;
esac

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing dataset: ${DATASET_PATH}" >&2
  exit 1
fi
META="${DATASET_PATH}.meta.json"
if [[ ! -f "${META}" ]]; then
  echo "[error] missing meta: ${META}" >&2
  exit 1
fi
MAX_PROMPT_LENGTH=$(python3 -c "import json; print(json.load(open('${META}'))['max_prompt_length'])")
COMPLETION_LENGTH=${COMPLETION_LENGTH:-1024}
if [[ "${COMPLETION_LENGTH}" != "1024" ]]; then
  RUN_NAME="${RUN_NAME}_c${COMPLETION_LENGTH}"
fi

export OPSD_SEQUENTIAL_DATA=1
export DATASET_PATH
export RUN_NAME
export MAX_PROMPT_LENGTH
export MAX_COMPLETION_LENGTH="${COMPLETION_LENGTH}"

echo "[aligned] spec=${SPEC} run=${RUN_NAME}"
echo "[aligned] dataset=${DATASET_PATH}"
echo "[aligned] max_prompt_length=${MAX_PROMPT_LENGTH} max_completion_length=${MAX_COMPLETION_LENGTH} sequential=1"
exec bash "${BASE_DIR}/${INNER}"
