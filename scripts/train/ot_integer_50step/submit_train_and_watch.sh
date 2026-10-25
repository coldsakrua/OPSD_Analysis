#!/bin/bash
# Submit 4 OT-integer 50-step OPSD trains + one CPU watcher.
#   qwen3-1.7b answer / solution
#   olmo3-7b-think answer / solution
# Watcher submits 4 think evals on checkpoint-50 for each run.
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "${BASE_DIR}"

TRAIN_17A=${TRAIN_17A:-scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_answer_50step.sh}
TRAIN_17S=${TRAIN_17S:-scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_solution_50step.sh}
TRAIN_OLA=${TRAIN_OLA:-scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_answer_50step.sh}
TRAIN_OLS=${TRAIN_OLS:-scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_solution_50step.sh}
WATCH_SCRIPT=${WATCH_SCRIPT:-scripts/train/ot_integer_50step/watch_then_eval.sh}

RUN_NAME_17A=${RUN_NAME_17A:-st_tt_clip005_1e_6_ot_integer_answer_50step_1p7b}
RUN_NAME_17S=${RUN_NAME_17S:-st_tt_clip005_1e_6_ot_integer_solution_50step_1p7b}
RUN_NAME_OLA=${RUN_NAME_OLA:-st_tt_clip005_1e_6_ot_integer_answer_50step_olmo7bt}
RUN_NAME_OLS=${RUN_NAME_OLS:-st_tt_clip005_1e_6_ot_integer_solution_50step_olmo7bt}
CKPT_STEP=${CKPT_STEP:-50}
EVAL_SEED=${EVAL_SEED:-42}

ANS_DS=${ANS_DS:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.correct.answer.sthink_tthink.integer_answer.sampler_seed1.n3200.parquet}
SOL_DS=${SOL_DS:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.opsd.solution.sthink_tthink.integer_answer.sampler_seed1.n3200.parquet}

for p in "${ANS_DS}" "${SOL_DS}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[error] missing dataset: ${p}" >&2
    echo "[error] run: python scripts/data/prepare_openthoughts_integer_n3200.py" >&2
    exit 1
  fi
done

mkdir -p log/train/1.7b log/train/olmo3-7b-think log/train/ot_integer_50step

out=$(sbatch --parsable "${TRAIN_17A}")
TRAIN_JID_17A=${out%%;*}
echo "[submit] 1.7b answer   jid=${TRAIN_JID_17A}  ${TRAIN_17A}"

out=$(sbatch --parsable "${TRAIN_17S}")
TRAIN_JID_17S=${out%%;*}
echo "[submit] 1.7b solution jid=${TRAIN_JID_17S}  ${TRAIN_17S}"

out=$(sbatch --parsable "${TRAIN_OLA}")
TRAIN_JID_OLA=${out%%;*}
echo "[submit] olmo answer   jid=${TRAIN_JID_OLA}  ${TRAIN_OLA}"

out=$(sbatch --parsable "${TRAIN_OLS}")
TRAIN_JID_OLS=${out%%;*}
echo "[submit] olmo solution jid=${TRAIN_JID_OLS}  ${TRAIN_OLS}"

watch_out=$(sbatch --parsable \
  --export=ALL,BASE_DIR="${BASE_DIR}",TRAIN_JID_17A="${TRAIN_JID_17A}",TRAIN_JID_17S="${TRAIN_JID_17S}",TRAIN_JID_OLA="${TRAIN_JID_OLA}",TRAIN_JID_OLS="${TRAIN_JID_OLS}",RUN_NAME_17A="${RUN_NAME_17A}",RUN_NAME_17S="${RUN_NAME_17S}",RUN_NAME_OLA="${RUN_NAME_OLA}",RUN_NAME_OLS="${RUN_NAME_OLS}",CKPT_STEP="${CKPT_STEP}",EVAL_SEED="${EVAL_SEED}" \
  "${WATCH_SCRIPT}")
WATCH_JID=${watch_out%%;*}

MANIFEST="${BASE_DIR}/log/train/ot_integer_50step/submit_latest.tsv"
{
  echo -e "key\tmodel_tag\trun_name\ttrain_jid\tscript"
  echo -e "1p7b_ans\tqwen3_1.7b\t${RUN_NAME_17A}\t${TRAIN_JID_17A}\t${TRAIN_17A}"
  echo -e "1p7b_sol\tqwen3_1.7b\t${RUN_NAME_17S}\t${TRAIN_JID_17S}\t${TRAIN_17S}"
  echo -e "olmo_ans\tolmo3_7b_think\t${RUN_NAME_OLA}\t${TRAIN_JID_OLA}\t${TRAIN_OLA}"
  echo -e "olmo_sol\tolmo3_7b_think\t${RUN_NAME_OLS}\t${TRAIN_JID_OLS}\t${TRAIN_OLS}"
  echo -e "watch\t-\t-\t${WATCH_JID}\t${WATCH_SCRIPT}"
} >"${MANIFEST}"

echo "[submit] watch_jid=${WATCH_JID} will eval checkpoint-${CKPT_STEP} (seed=${EVAL_SEED})"
echo "[submit] answer_ds=${ANS_DS}"
echo "[submit] solution_ds=${SOL_DS}"
echo "[submit] manifest=${MANIFEST}"
echo "[submit] watch log: log/train/ot_integer_50step/watch_slurm.${WATCH_JID}.out"
