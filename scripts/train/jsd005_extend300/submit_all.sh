#!/bin/bash
# Submit: (1) existing checkpoint-50 evals, (2) resume trains 100→300, (3) CPU watcher.
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "${BASE_DIR}"
mkdir -p \
  log/train/jsd005_extend300 \
  log/train/1.7b \
  log/train/olmo3-7b-think \
  log/eval/1.7b/array \
  log/eval/olmo_7b_think/array

DIR="${BASE_DIR}/scripts/train/jsd005_extend300"
SUBMIT="${DIR}/submit_four_think_evals.sh"
chmod +x \
  "${DIR}/opsd_1p7b_resume_to_300.sh" \
  "${DIR}/opsd_olmo7bt_resume_to_300.sh" \
  "${SUBMIT}" \
  "${DIR}/watch_then_eval.sh" \
  "${DIR}/sbatch_watch.sh" \
  "${BASE_DIR}/scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts.sh" \
  "${BASE_DIR}/scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts.sh"

CKPT50_1P7B="${BASE_DIR}/outputs/qwen3_1.7b/st_tt_clip005_1e_6_openthoughts_1p7b/3212996/checkpoint-50"
CKPT50_OLMO="${BASE_DIR}/outputs/olmo3_7b_think/st_tt_clip005_1e_6_openthoughts_olmo7bt/3233311/checkpoint-50"

echo "=== [1/3] submit checkpoint-50 think evals (array) ==="
MODEL_KEY=qwen3_1.7b CHECKPOINT_PATH="${CKPT50_1P7B}" EVAL_TAG=st_tt_clip005_1e6_ot_1p7b_ckpt50 \
  SEED=42 bash "${SUBMIT}"
MODEL_KEY=olmo3_7b_think CHECKPOINT_PATH="${CKPT50_OLMO}" EVAL_TAG=st_tt_clip005_1e6_olmo7bt_ckpt50 \
  SEED=42 bash "${SUBMIT}"

echo "=== [2/3] submit resume trains (100 → 300) ==="
# Always export BASE_DIR: Slurm spool copies break BASH_SOURCE-based path resolution.
jid_1p7b=$(sbatch --parsable --export=ALL,BASE_DIR="${BASE_DIR}" "${DIR}/opsd_1p7b_resume_to_300.sh")
jid_1p7b="${jid_1p7b%%;*}"
echo "[submit] 1.7b resume jid=${jid_1p7b}"

jid_olmo=$(sbatch --parsable --export=ALL,BASE_DIR="${BASE_DIR}" "${DIR}/opsd_olmo7bt_resume_to_300.sh")
jid_olmo="${jid_olmo%%;*}"
echo "[submit] olmo7bt resume jid=${jid_olmo}"

MANIFEST="${BASE_DIR}/log/train/jsd005_extend300/submit_latest.tsv"
{
  echo -e "# model_key\tmodel_tag\trun_name\toutput_jid\ttrain_jid\teval_steps"
  echo -e "qwen3_1.7b\tqwen3_1.7b\tst_tt_clip005_1e_6_openthoughts_1p7b\t3212996\t${jid_1p7b}\t150,200,250,300"
  echo -e "olmo3_7b_think\tolmo3_7b_think\tst_tt_clip005_1e_6_openthoughts_olmo7bt\t3233311\t${jid_olmo}\t150,200,250,300"
} >"${MANIFEST}"
echo "[submit] wrote ${MANIFEST}"

echo "=== [3/3] submit CPU watcher ==="
watch_jid=$(sbatch --parsable --export=ALL,BASE_DIR="${BASE_DIR}",MANIFEST="${MANIFEST}" \
  "${DIR}/sbatch_watch.sh")
watch_jid="${watch_jid%%;*}"
echo "[submit] watcher jid=${watch_jid}"

REPORT="${BASE_DIR}/log/train/jsd005_extend300/submit_report.${jid_1p7b}_${jid_olmo}.txt"
{
  echo "submitted_at=$(date -Is)"
  echo "ckpt50_eval_1p7b=st_tt_clip005_1e6_ot_1p7b_ckpt50"
  echo "ckpt50_eval_olmo=st_tt_clip005_1e6_olmo7bt_ckpt50"
  echo "train_1p7b=${jid_1p7b}"
  echo "train_olmo=${jid_olmo}"
  echo "watch=${watch_jid}"
  echo "manifest=${MANIFEST}"
} | tee "${REPORT}"

echo "[done] report=${REPORT}"
