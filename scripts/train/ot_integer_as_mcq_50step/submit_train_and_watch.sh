#!/bin/bash
# Submit OT integer→MCQ answer-only trains (letter / letter+option) × 2 models + watcher.
set -euo pipefail
BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "${BASE_DIR}"

TRAIN_17L=${TRAIN_17L:-scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_as_mcq_letter_50step.sh}
TRAIN_17O=${TRAIN_17O:-scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_as_mcq_letter_option_50step.sh}
TRAIN_OLL=${TRAIN_OLL:-scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_as_mcq_letter_50step.sh}
TRAIN_OLO=${TRAIN_OLO:-scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_as_mcq_letter_option_50step.sh}
WATCH_SCRIPT=${WATCH_SCRIPT:-scripts/train/ot_integer_as_mcq_50step/watch_then_eval.sh}

RUN_NAME_17L=${RUN_NAME_17L:-st_tt_clip005_1e_6_ot_integer_as_mcq_letter_50step_1p7b}
RUN_NAME_17O=${RUN_NAME_17O:-st_tt_clip005_1e_6_ot_integer_as_mcq_letter_option_50step_1p7b}
RUN_NAME_OLL=${RUN_NAME_OLL:-st_tt_clip005_1e_6_ot_integer_as_mcq_letter_50step_olmo7bt}
RUN_NAME_OLO=${RUN_NAME_OLO:-st_tt_clip005_1e_6_ot_integer_as_mcq_letter_option_50step_olmo7bt}
CKPT_STEP=${CKPT_STEP:-50}
EVAL_SEED=${EVAL_SEED:-42}

DS_L=${DS_L:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.correct.answer.sthink_tthink.integer_answer.as_mcq_letter.sampler_seed1.n3200.parquet}
DS_O=${DS_O:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.correct.answer.sthink_tthink.integer_answer.as_mcq_letter_option.sampler_seed1.n3200.parquet}
for p in "${DS_L}" "${DS_O}"; do
  if [[ ! -f "${p}" ]]; then
    echo "[error] missing ${p}" >&2
    echo "[error] run: python scripts/data/prepare_openthoughts_integer_as_mcq_n3200.py" >&2
    exit 1
  fi
done

mkdir -p log/train/1.7b log/train/olmo3-7b-think log/train/ot_integer_as_mcq_50step

out=$(sbatch --parsable "${TRAIN_17L}"); TRAIN_JID_17L=${out%%;*}; echo "[submit] 1.7b mcq_letter jid=${TRAIN_JID_17L}"
out=$(sbatch --parsable "${TRAIN_17O}"); TRAIN_JID_17O=${out%%;*}; echo "[submit] 1.7b mcq_letter_option jid=${TRAIN_JID_17O}"
out=$(sbatch --parsable "${TRAIN_OLL}"); TRAIN_JID_OLL=${out%%;*}; echo "[submit] olmo mcq_letter jid=${TRAIN_JID_OLL}"
out=$(sbatch --parsable "${TRAIN_OLO}"); TRAIN_JID_OLO=${out%%;*}; echo "[submit] olmo mcq_letter_option jid=${TRAIN_JID_OLO}"

watch_out=$(sbatch --parsable \
  --export=ALL,BASE_DIR="${BASE_DIR}",TRAIN_JID_17L="${TRAIN_JID_17L}",TRAIN_JID_17O="${TRAIN_JID_17O}",TRAIN_JID_OLL="${TRAIN_JID_OLL}",TRAIN_JID_OLO="${TRAIN_JID_OLO}",RUN_NAME_17L="${RUN_NAME_17L}",RUN_NAME_17O="${RUN_NAME_17O}",RUN_NAME_OLL="${RUN_NAME_OLL}",RUN_NAME_OLO="${RUN_NAME_OLO}",CKPT_STEP="${CKPT_STEP}",EVAL_SEED="${EVAL_SEED}" \
  "${WATCH_SCRIPT}")
WATCH_JID=${watch_out%%;*}

MANIFEST="${BASE_DIR}/log/train/ot_integer_as_mcq_50step/submit_latest.tsv"
{
  echo -e "key\tmodel_tag\trun_name\ttrain_jid\tscript"
  echo -e "1p7b_mcq_let\tqwen3_1.7b\t${RUN_NAME_17L}\t${TRAIN_JID_17L}\t${TRAIN_17L}"
  echo -e "1p7b_mcq_opt\tqwen3_1.7b\t${RUN_NAME_17O}\t${TRAIN_JID_17O}\t${TRAIN_17O}"
  echo -e "olmo_mcq_let\tolmo3_7b_think\t${RUN_NAME_OLL}\t${TRAIN_JID_OLL}\t${TRAIN_OLL}"
  echo -e "olmo_mcq_opt\tolmo3_7b_think\t${RUN_NAME_OLO}\t${TRAIN_JID_OLO}\t${TRAIN_OLO}"
  echo -e "watch\t-\t-\t${WATCH_JID}\t${WATCH_SCRIPT}"
} >"${MANIFEST}"
echo "[submit] watch_jid=${WATCH_JID} manifest=${MANIFEST}"
