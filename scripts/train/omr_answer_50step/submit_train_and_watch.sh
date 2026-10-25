#!/bin/bash
# Submit Qwen3-1.7B + 4B OMR-answer st_tt 50-step trains + joint CPU watcher.
# Privilege/data = OMR answer-only; batch/hparams follow each model's own st_tt defaults
# (1.7b: gbs=64; 4b: gbs=64). Watcher submits 4 think evals on ckpt-50.
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "${BASE_DIR}"

TRAIN_17=${TRAIN_17:-scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_omr_answer_50step.sh}
TRAIN_4=${TRAIN_4:-scripts/train/qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_omr_answer_50step.sh}
WATCH_SCRIPT=${WATCH_SCRIPT:-scripts/train/omr_answer_50step/watch_then_eval.sh}
RUN_NAME_17=${RUN_NAME_17:-st_tt_clip005_1e_6_omr_answer_50step_1p7b}
RUN_NAME_4=${RUN_NAME_4:-st_tt_clip005_1e_6_omr_answer_50step_4b}
CKPT_STEP=${CKPT_STEP:-50}

DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openmathreasoning/preprocessed/omr.correct.answer.sthink_tthink.maxprompt1032.grpo_order.parquet}
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing OMR answer parquet: ${DATASET_PATH}" >&2
  echo "[error] run: sbatch scripts/data/run_preprocess_opsd_omr_integer_answer.sh" >&2
  exit 1
fi

mkdir -p log/train/1.7b log/train/4b log/train/omr_answer_50step

out17=$(sbatch --parsable "${TRAIN_17}")
TRAIN_JID_17=${out17%%;*}
echo "[submit] 1.7b train_jid=${TRAIN_JID_17} script=${TRAIN_17}"

out4=$(sbatch --parsable "${TRAIN_4}")
TRAIN_JID_4=${out4%%;*}
echo "[submit] 4b   train_jid=${TRAIN_JID_4} script=${TRAIN_4}"

watch_out=$(sbatch --parsable \
  --export=ALL,BASE_DIR="${BASE_DIR}",TRAIN_JID_17="${TRAIN_JID_17}",TRAIN_JID_4="${TRAIN_JID_4}",RUN_NAME_17="${RUN_NAME_17}",RUN_NAME_4="${RUN_NAME_4}",CKPT_STEP="${CKPT_STEP}" \
  "${WATCH_SCRIPT}")
WATCH_JID=${watch_out%%;*}
echo "[submit] watch_jid=${WATCH_JID} will eval on checkpoint-${CKPT_STEP}"
echo "[submit] dataset=${DATASET_PATH}"
echo "[submit] 1.7b log: log/train/1.7b/opsd_st_tt_clip005_1e6_omr_answer_1p7b.${TRAIN_JID_17}.out"
echo "[submit] 4b   log: log/train/4b/opsd_st_tt_clip005_1e6_omr_answer_4b.${TRAIN_JID_4}.out"
echo "[submit] watch log: log/train/omr_answer_50step/watch_slurm.${WATCH_JID}.out"
