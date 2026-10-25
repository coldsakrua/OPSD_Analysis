#!/bin/bash
# Submit Qwen3-4B OMR-answer st_tt 50-step train + CPU watcher (evals on ckpt-50).
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "${BASE_DIR}"

TRAIN_SCRIPT=${TRAIN_SCRIPT:-scripts/train/qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_omr_answer_50step.sh}
WATCH_SCRIPT=${WATCH_SCRIPT:-scripts/train/omr_answer_50step/watch_then_eval_4b.sh}
RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_omr_answer_50step_4b}
MODEL_TAG=${MODEL_TAG:-qwen3_4b}
CKPT_STEP=${CKPT_STEP:-50}

DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openmathreasoning/preprocessed/omr.correct.answer.sthink_tthink.maxprompt1032.grpo_order.parquet}
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing OMR answer parquet: ${DATASET_PATH}" >&2
  echo "[error] run: sbatch scripts/data/run_preprocess_opsd_omr_integer_answer.sh" >&2
  exit 1
fi

mkdir -p log/train/4b log/train/omr_answer_50step

train_out=$(sbatch --parsable "${TRAIN_SCRIPT}")
TRAIN_JID=${train_out%%;*}
echo "[submit] train_jid=${TRAIN_JID} script=${TRAIN_SCRIPT}"

watch_out=$(sbatch --parsable \
  --export=ALL,BASE_DIR="${BASE_DIR}",TRAIN_JID="${TRAIN_JID}",RUN_NAME="${RUN_NAME}",MODEL_TAG="${MODEL_TAG}",CKPT_STEP="${CKPT_STEP}" \
  "${WATCH_SCRIPT}")
WATCH_JID=${watch_out%%;*}
echo "[submit] watch_jid=${WATCH_JID} will eval on checkpoint-${CKPT_STEP}"
echo "[submit] dataset=${DATASET_PATH}"
echo "[submit] train log: log/train/4b/opsd_st_tt_clip005_1e6_omr_answer_4b.${TRAIN_JID}.out"
echo "[submit] watch log: log/train/omr_answer_50step/watch_4b_slurm.${WATCH_JID}.out"
