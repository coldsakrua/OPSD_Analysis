#!/bin/bash
# Submit 2 OT-integer **rewritten-answer** 50-step OPSD trains + one CPU watcher.
#   qwen3-1.7b answer (rewritten frac/decimal)
#   olmo3-7b-think answer (rewritten frac/decimal)
# Watcher submits 4 think evals on checkpoint-50 for each run.
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "${BASE_DIR}"

TRAIN_17A=${TRAIN_17A:-scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_answer_rewritten_50step.sh}
TRAIN_OLA=${TRAIN_OLA:-scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_answer_rewritten_50step.sh}
WATCH_SCRIPT=${WATCH_SCRIPT:-scripts/train/ot_integer_answer_rewritten_50step/watch_then_eval.sh}

RUN_NAME_17A=${RUN_NAME_17A:-st_tt_clip005_1e_6_ot_integer_answer_rewritten_50step_1p7b}
RUN_NAME_OLA=${RUN_NAME_OLA:-st_tt_clip005_1e_6_ot_integer_answer_rewritten_50step_olmo7bt}
CKPT_STEP=${CKPT_STEP:-50}
EVAL_SEED=${EVAL_SEED:-42}

ANS_DS=${ANS_DS:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.correct.answer.sthink_tthink.integer_answer.rewritten_frac_decimal.sampler_seed1.n3200.parquet}

if [[ ! -f "${ANS_DS}" ]]; then
  echo "[error] missing dataset: ${ANS_DS}" >&2
  echo "[error] run: python scripts/data/prepare_openthoughts_integer_answer_rewritten_n3200.py" >&2
  exit 1
fi

mkdir -p log/train/1.7b log/train/olmo3-7b-think log/train/ot_integer_answer_rewritten_50step

out=$(sbatch --parsable "${TRAIN_17A}")
TRAIN_JID_17A=${out%%;*}
echo "[submit] 1.7b answer_rewritten jid=${TRAIN_JID_17A}  ${TRAIN_17A}"

out=$(sbatch --parsable "${TRAIN_OLA}")
TRAIN_JID_OLA=${out%%;*}
echo "[submit] olmo answer_rewritten jid=${TRAIN_JID_OLA}  ${TRAIN_OLA}"

watch_out=$(sbatch --parsable \
  --export=ALL,BASE_DIR="${BASE_DIR}",TRAIN_JID_17A="${TRAIN_JID_17A}",TRAIN_JID_OLA="${TRAIN_JID_OLA}",RUN_NAME_17A="${RUN_NAME_17A}",RUN_NAME_OLA="${RUN_NAME_OLA}",CKPT_STEP="${CKPT_STEP}",EVAL_SEED="${EVAL_SEED}" \
  "${WATCH_SCRIPT}")
WATCH_JID=${watch_out%%;*}

MANIFEST="${BASE_DIR}/log/train/ot_integer_answer_rewritten_50step/submit_latest.tsv"
{
  echo -e "key\tmodel_tag\trun_name\ttrain_jid\tscript"
  echo -e "1p7b_ans_rw\tqwen3_1.7b\t${RUN_NAME_17A}\t${TRAIN_JID_17A}\t${TRAIN_17A}"
  echo -e "olmo_ans_rw\tolmo3_7b_think\t${RUN_NAME_OLA}\t${TRAIN_JID_OLA}\t${TRAIN_OLA}"
  echo -e "watch\t-\t-\t${WATCH_JID}\t${WATCH_SCRIPT}"
} >"${MANIFEST}"

echo "[submit] watch_jid=${WATCH_JID} will eval checkpoint-${CKPT_STEP} (seed=${EVAL_SEED})"
echo "[submit] answer_rewritten_ds=${ANS_DS}"
echo "[submit] manifest=${MANIFEST}"
echo "[submit] watch log: log/train/ot_integer_answer_rewritten_50step/watch_slurm.${WATCH_JID}.out"
