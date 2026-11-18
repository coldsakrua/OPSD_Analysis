#!/bin/bash
# Submit 8 OT-integer **solution** answer-variant 50-step trains + one CPU watcher.
# Variants (same 3200 problems/order):
#   solution | rewritten final boxed | wrong_n10p1 | wrong_rand50
# × qwen3-1.7b + olmo3-7b-think
# Watcher is submitted AFTER the last train; it evals checkpoint-50 for each run.
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "${BASE_DIR}"

TRAIN_17S=${TRAIN_17S:-scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_solution_50step.sh}
TRAIN_17R=${TRAIN_17R:-scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_solution_rewritten_50step.sh}
TRAIN_17W=${TRAIN_17W:-scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_solution_wrong_50step.sh}
TRAIN_17X=${TRAIN_17X:-scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_solution_wrong_rand50_50step.sh}
TRAIN_OLS=${TRAIN_OLS:-scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_solution_50step.sh}
TRAIN_OLR=${TRAIN_OLR:-scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_solution_rewritten_50step.sh}
TRAIN_OLW=${TRAIN_OLW:-scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_solution_wrong_50step.sh}
TRAIN_OLX=${TRAIN_OLX:-scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_solution_wrong_rand50_50step.sh}
WATCH_SCRIPT=${WATCH_SCRIPT:-scripts/train/ot_integer_solution_answer_variants_50step/watch_then_eval.sh}

RUN_NAME_17S=${RUN_NAME_17S:-st_tt_clip005_1e_6_ot_integer_solution_50step_1p7b}
RUN_NAME_17R=${RUN_NAME_17R:-st_tt_clip005_1e_6_ot_integer_solution_rewritten_50step_1p7b}
RUN_NAME_17W=${RUN_NAME_17W:-st_tt_clip005_1e_6_ot_integer_solution_wrong_50step_1p7b}
RUN_NAME_17X=${RUN_NAME_17X:-st_tt_clip005_1e_6_ot_integer_solution_wrong_rand50_50step_1p7b}
RUN_NAME_OLS=${RUN_NAME_OLS:-st_tt_clip005_1e_6_ot_integer_solution_50step_olmo7bt}
RUN_NAME_OLR=${RUN_NAME_OLR:-st_tt_clip005_1e_6_ot_integer_solution_rewritten_50step_olmo7bt}
RUN_NAME_OLW=${RUN_NAME_OLW:-st_tt_clip005_1e_6_ot_integer_solution_wrong_50step_olmo7bt}
RUN_NAME_OLX=${RUN_NAME_OLX:-st_tt_clip005_1e_6_ot_integer_solution_wrong_rand50_50step_olmo7bt}
CKPT_STEP=${CKPT_STEP:-50}
EVAL_SEED=${EVAL_SEED:-42}

PRE=${BASE_DIR}/data/openthoughts/preprocessed
for ds in \
  "${PRE}/openthoughts.opsd.solution.sthink_tthink.integer_answer.sampler_seed1.n3200.parquet" \
  "${PRE}/openthoughts.opsd.solution.sthink_tthink.integer_answer.rewritten_frac_decimal.sampler_seed1.n3200.parquet" \
  "${PRE}/openthoughts.opsd.solution.sthink_tthink.integer_answer.wrong_n10p1.sampler_seed1.n3200.parquet" \
  "${PRE}/openthoughts.opsd.solution.sthink_tthink.integer_answer.wrong_rand50.sampler_seed1.n3200.parquet"
do
  if [[ ! -f "${ds}" ]]; then
    echo "[error] missing dataset: ${ds}" >&2
    echo "[error] run: sbatch scripts/data/run_prepare_ot_integer_solution_answer_variants_n3200.sh" >&2
    exit 1
  fi
done

mkdir -p log/train/1.7b log/train/olmo3-7b-think log/train/ot_integer_solution_answer_variants_50step

out=$(sbatch --parsable "${TRAIN_17S}"); TRAIN_JID_17S=${out%%;*}; echo "[submit] 1.7b solution          jid=${TRAIN_JID_17S}"
out=$(sbatch --parsable "${TRAIN_17R}"); TRAIN_JID_17R=${out%%;*}; echo "[submit] 1.7b sol_rewritten    jid=${TRAIN_JID_17R}"
out=$(sbatch --parsable "${TRAIN_17W}"); TRAIN_JID_17W=${out%%;*}; echo "[submit] 1.7b sol_wrong_n10p1  jid=${TRAIN_JID_17W}"
out=$(sbatch --parsable "${TRAIN_17X}"); TRAIN_JID_17X=${out%%;*}; echo "[submit] 1.7b sol_wrong_rand50 jid=${TRAIN_JID_17X}"
out=$(sbatch --parsable "${TRAIN_OLS}"); TRAIN_JID_OLS=${out%%;*}; echo "[submit] olmo solution          jid=${TRAIN_JID_OLS}"
out=$(sbatch --parsable "${TRAIN_OLR}"); TRAIN_JID_OLR=${out%%;*}; echo "[submit] olmo sol_rewritten    jid=${TRAIN_JID_OLR}"
out=$(sbatch --parsable "${TRAIN_OLW}"); TRAIN_JID_OLW=${out%%;*}; echo "[submit] olmo sol_wrong_n10p1  jid=${TRAIN_JID_OLW}"
out=$(sbatch --parsable "${TRAIN_OLX}"); TRAIN_JID_OLX=${out%%;*}; echo "[submit] olmo sol_wrong_rand50 jid=${TRAIN_JID_OLX}  (last train)"

# Watcher submitted after the last train.
watch_out=$(sbatch --parsable \
  --export=ALL,BASE_DIR="${BASE_DIR}",\
TRAIN_JID_17S="${TRAIN_JID_17S}",TRAIN_JID_17R="${TRAIN_JID_17R}",TRAIN_JID_17W="${TRAIN_JID_17W}",TRAIN_JID_17X="${TRAIN_JID_17X}",\
TRAIN_JID_OLS="${TRAIN_JID_OLS}",TRAIN_JID_OLR="${TRAIN_JID_OLR}",TRAIN_JID_OLW="${TRAIN_JID_OLW}",TRAIN_JID_OLX="${TRAIN_JID_OLX}",\
RUN_NAME_17S="${RUN_NAME_17S}",RUN_NAME_17R="${RUN_NAME_17R}",RUN_NAME_17W="${RUN_NAME_17W}",RUN_NAME_17X="${RUN_NAME_17X}",\
RUN_NAME_OLS="${RUN_NAME_OLS}",RUN_NAME_OLR="${RUN_NAME_OLR}",RUN_NAME_OLW="${RUN_NAME_OLW}",RUN_NAME_OLX="${RUN_NAME_OLX}",\
CKPT_STEP="${CKPT_STEP}",EVAL_SEED="${EVAL_SEED}" \
  "${WATCH_SCRIPT}")
WATCH_JID=${watch_out%%;*}

MANIFEST="${BASE_DIR}/log/train/ot_integer_solution_answer_variants_50step/submit_latest.tsv"
{
  echo -e "key\tmodel_tag\trun_name\ttrain_jid\tscript"
  echo -e "1p7b_sol\tqwen3_1.7b\t${RUN_NAME_17S}\t${TRAIN_JID_17S}\t${TRAIN_17S}"
  echo -e "1p7b_sol_rw\tqwen3_1.7b\t${RUN_NAME_17R}\t${TRAIN_JID_17R}\t${TRAIN_17R}"
  echo -e "1p7b_sol_wrong\tqwen3_1.7b\t${RUN_NAME_17W}\t${TRAIN_JID_17W}\t${TRAIN_17W}"
  echo -e "1p7b_sol_wr50\tqwen3_1.7b\t${RUN_NAME_17X}\t${TRAIN_JID_17X}\t${TRAIN_17X}"
  echo -e "olmo_sol\tolmo3_7b_think\t${RUN_NAME_OLS}\t${TRAIN_JID_OLS}\t${TRAIN_OLS}"
  echo -e "olmo_sol_rw\tolmo3_7b_think\t${RUN_NAME_OLR}\t${TRAIN_JID_OLR}\t${TRAIN_OLR}"
  echo -e "olmo_sol_wrong\tolmo3_7b_think\t${RUN_NAME_OLW}\t${TRAIN_JID_OLW}\t${TRAIN_OLW}"
  echo -e "olmo_sol_wr50\tolmo3_7b_think\t${RUN_NAME_OLX}\t${TRAIN_JID_OLX}\t${TRAIN_OLX}"
  echo -e "watch\t-\t-\t${WATCH_JID}\t${WATCH_SCRIPT}"
} >"${MANIFEST}"

echo "[submit] watch_jid=${WATCH_JID} will eval checkpoint-${CKPT_STEP} (seed=${EVAL_SEED})"
echo "[submit] manifest=${MANIFEST}"
echo "[submit] watch log: log/train/ot_integer_solution_answer_variants_50step/watch_slurm.${WATCH_JID}.out"
