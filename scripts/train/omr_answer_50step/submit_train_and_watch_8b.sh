#!/bin/bash
# Submit Qwen3-8B OMR-answer st_tt 50-step train + CPU watcher (evals on ckpt-50).
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "${BASE_DIR}"

TRAIN_SCRIPT=${TRAIN_SCRIPT:-scripts/train/qwen3_8b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_omr_answer_50step.sh}
WATCH_SCRIPT=${WATCH_SCRIPT:-scripts/train/omr_answer_50step/watch_then_eval_8b.sh}
RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_omr_answer_50step_8b}
MODEL_TAG=${MODEL_TAG:-qwen3_8b}
CKPT_STEP=${CKPT_STEP:-50}

mkdir -p log/train/8b log/train/omr_answer_50step

train_out=$(sbatch --parsable "${TRAIN_SCRIPT}")
TRAIN_JID=${train_out%%;*}
echo "[submit] train_jid=${TRAIN_JID} script=${TRAIN_SCRIPT}"

watch_out=$(sbatch --parsable \
  --export=ALL,BASE_DIR="${BASE_DIR}",TRAIN_JID="${TRAIN_JID}",RUN_NAME="${RUN_NAME}",MODEL_TAG="${MODEL_TAG}",CKPT_STEP="${CKPT_STEP}" \
  "${WATCH_SCRIPT}")
WATCH_JID=${watch_out%%;*}
echo "[submit] watch_jid=${WATCH_JID} will eval on checkpoint-${CKPT_STEP}"
echo "[submit] train log: log/train/8b/opsd_st_tt_clip005_1e6_omr_answer_8b.${TRAIN_JID}.out"
echo "[submit] watch log: log/train/omr_answer_50step/watch_8b_slurm.${WATCH_JID}.out"
