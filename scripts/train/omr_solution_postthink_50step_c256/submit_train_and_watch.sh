#!/bin/bash
# Submit Qwen3-1.7B / 4B / 8B OMR-solution (postthink) st_tt 50-step c256 trains
# (max_completion=256) + joint CPU watcher for aime24/25/26 + hmmt25 @ ckpt-50.
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "${BASE_DIR}"

TRAIN_17=${TRAIN_17:-scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_omr_solution_postthink_50step_c256.sh}
TRAIN_4=${TRAIN_4:-scripts/train/qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_omr_solution_postthink_50step_c256.sh}
TRAIN_8=${TRAIN_8:-scripts/train/qwen3_8b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_omr_solution_postthink_50step_c256.sh}
WATCH_SCRIPT=${WATCH_SCRIPT:-scripts/train/omr_solution_postthink_50step_c256/watch_then_eval.sh}
RUN_NAME_17=${RUN_NAME_17:-st_tt_clip005_1e_6_omr_solution_postthink_50step_1p7b_c256}
RUN_NAME_4=${RUN_NAME_4:-st_tt_clip005_1e_6_omr_solution_postthink_50step_4b_c256}
RUN_NAME_8=${RUN_NAME_8:-st_tt_clip005_1e_6_omr_solution_postthink_50step_8b_c256}
CKPT_STEP=${CKPT_STEP:-50}

DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openmathreasoning/preprocessed/omr.opsd.solution.postthink.sthink_tthink.grpo_sampler_seed1.n3200.parquet}
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing GRPO-aligned solution parquet: ${DATASET_PATH}" >&2
  echo "[error] run: sbatch scripts/data/run_preprocess_opsd_omr_integer_solution_postthink.sh" >&2
  exit 1
fi

mkdir -p log/train/{1.7b,4b,8b} log/train/omr_solution_postthink_50step_c256
chmod +x "${TRAIN_17}" "${TRAIN_4}" "${TRAIN_8}" "${WATCH_SCRIPT}"

out17=$(sbatch --parsable "${TRAIN_17}")
TRAIN_JID_17=${out17%%;*}
echo "[submit] 1.7b train_jid=${TRAIN_JID_17} script=${TRAIN_17}"

out4=$(sbatch --parsable "${TRAIN_4}")
TRAIN_JID_4=${out4%%;*}
echo "[submit] 4b   train_jid=${TRAIN_JID_4} script=${TRAIN_4}"

out8=$(sbatch --parsable "${TRAIN_8}")
TRAIN_JID_8=${out8%%;*}
echo "[submit] 8b   train_jid=${TRAIN_JID_8} script=${TRAIN_8}"

watch_out=$(sbatch --parsable \
  --export=ALL,BASE_DIR="${BASE_DIR}",TRAIN_JID_17="${TRAIN_JID_17}",TRAIN_JID_4="${TRAIN_JID_4}",TRAIN_JID_8="${TRAIN_JID_8}",RUN_NAME_17="${RUN_NAME_17}",RUN_NAME_4="${RUN_NAME_4}",RUN_NAME_8="${RUN_NAME_8}",CKPT_STEP="${CKPT_STEP}" \
  "${WATCH_SCRIPT}")
WATCH_JID=${watch_out%%;*}
echo "[submit] watch_jid=${WATCH_JID} will eval on checkpoint-${CKPT_STEP}"
echo "[submit] dataset=${DATASET_PATH}"
echo "[submit] completion=256 (c256)"
echo "[submit] 1.7b log: log/train/1.7b/opsd_st_tt_omr_sol_c256_1p7b.${TRAIN_JID_17}.out"
echo "[submit] 4b   log: log/train/4b/opsd_st_tt_omr_sol_c256_4b.${TRAIN_JID_4}.out"
echo "[submit] 8b   log: log/train/8b/opsd_st_tt_omr_sol_c256_8b.${TRAIN_JID_8}.out"
echo "[submit] watch log: log/train/omr_solution_postthink_50step_c256/watch_slurm.${WATCH_JID}.out"
