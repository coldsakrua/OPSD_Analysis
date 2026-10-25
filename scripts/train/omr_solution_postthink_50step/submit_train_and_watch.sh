#!/bin/bash
# Submit Qwen3-1.7B + 4B OMR-solution (postthink) st_tt 50-step trains + joint CPU watcher.
# Dataset = GRPO RandomSampler(seed=1) first 3200 prompts with gold solution privilege.
# Watcher submits 4 think evals (aime24/25/26 + hmmt25) when ckpt-50 is ready.
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "${BASE_DIR}"

TRAIN_17=${TRAIN_17:-scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_omr_solution_postthink_50step.sh}
TRAIN_4=${TRAIN_4:-scripts/train/qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_omr_solution_postthink_50step.sh}
WATCH_SCRIPT=${WATCH_SCRIPT:-scripts/train/omr_solution_postthink_50step/watch_then_eval.sh}
RUN_NAME_17=${RUN_NAME_17:-st_tt_clip005_1e_6_omr_solution_postthink_50step_1p7b}
RUN_NAME_4=${RUN_NAME_4:-st_tt_clip005_1e_6_omr_solution_postthink_50step_4b}
CKPT_STEP=${CKPT_STEP:-50}

DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openmathreasoning/preprocessed/omr.opsd.solution.postthink.sthink_tthink.grpo_sampler_seed1.n3200.parquet}
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing GRPO-aligned solution parquet: ${DATASET_PATH}" >&2
  echo "[error] run: sbatch scripts/data/run_preprocess_opsd_omr_integer_solution_postthink.sh" >&2
  exit 1
fi

mkdir -p log/train/1.7b log/train/4b log/train/omr_solution_postthink_50step

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
echo "[submit] 1.7b log: log/train/1.7b/opsd_st_tt_clip005_1e6_omr_sol_1p7b.${TRAIN_JID_17}.out"
echo "[submit] 4b   log: log/train/4b/opsd_st_tt_clip005_1e6_omr_sol_4b.${TRAIN_JID_4}.out"
echo "[submit] watch log: log/train/omr_solution_postthink_50step/watch_slurm.${WATCH_JID}.out"
