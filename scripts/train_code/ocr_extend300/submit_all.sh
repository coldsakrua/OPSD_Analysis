#!/bin/bash
# Submit OCR OPSD resume trains (100 → 300) for 1.7B + Olmo7B-Think, then a
# multi-step CPU watcher that submits LCB / HumanEval+ / MBPP+ at 150/200/250/300.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train_code/ocr_extend300/submit_all.sh
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "${BASE_DIR}"
mkdir -p \
  log/train_code/ocr_extend300 \
  log/train_code/1.7b \
  log/train_code/olmo3-7b-think

DIR="${BASE_DIR}/scripts/train_code/ocr_extend300"
chmod +x \
  "${DIR}/opsd_1p7b_resume_to_300.sh" \
  "${DIR}/opsd_olmo7bt_resume_to_300.sh" \
  "${DIR}/watch_then_eval_multi.sh" \
  "${DIR}/sbatch_watch_multi.sh" \
  "${BASE_DIR}/scripts/train_code/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh" \
  "${BASE_DIR}/scripts/train_code/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh" \
  "${BASE_DIR}/scripts/train_code/ocr/submit_three_code_evals.sh"

CKPT_1P7B="${BASE_DIR}/outputs/qwen3_1.7b/st_tt_clip005_1e_6_opencodereasoning_1p7b/3908495/checkpoint-100"
CKPT_OLMO="${BASE_DIR}/outputs/olmo3_7b_think/st_tt_clip005_1e_6_opencodereasoning_olmo7bt/3908500/checkpoint-100"
for p in "${CKPT_1P7B}" "${CKPT_OLMO}"; do
  if [[ ! -f "${p}/trainer_state.json" ]]; then
    echo "[error] missing resume ckpt: ${p}" >&2
    exit 1
  fi
done

EVAL_STEPS=${EVAL_STEPS:-150,200,250,300}
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${BASE_DIR}/log/train_code/ocr_extend300/submit_${STAMP}.tsv"
MANIFEST="${BASE_DIR}/log/train_code/ocr_extend300/submit_latest.tsv"

echo "=== submit resume trains (checkpoint-100 → max_steps=300, save_steps=25) ==="
jid_1p7b=$(sbatch --parsable --chdir="${BASE_DIR}" --export=ALL,BASE_DIR="${BASE_DIR}" \
  "${DIR}/opsd_1p7b_resume_to_300.sh")
jid_1p7b="${jid_1p7b%%;*}"
echo "[submit] 1.7b resume jid=${jid_1p7b} output_dir=.../3908495"

jid_olmo=$(sbatch --parsable --chdir="${BASE_DIR}" --export=ALL,BASE_DIR="${BASE_DIR}" \
  "${DIR}/opsd_olmo7bt_resume_to_300.sh")
jid_olmo="${jid_olmo%%;*}"
echo "[submit] olmo7bt resume jid=${jid_olmo} output_dir=.../3908500"

{
  echo -e "# model_key\tmodel_tag\trun_name\toutput_jid\ttrain_jid\teval_steps"
  echo -e "qwen3_1.7b\tqwen3_1.7b\tst_tt_clip005_1e_6_opencodereasoning_1p7b\t3908495\t${jid_1p7b}\t${EVAL_STEPS}"
  echo -e "olmo3_7b_think\tolmo3_7b_think\tst_tt_clip005_1e_6_opencodereasoning_olmo7bt\t3908500\t${jid_olmo}\t${EVAL_STEPS}"
} >"${MANIFEST_STAMPED}"
cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"

watch_jid=$(sbatch --parsable --chdir="${BASE_DIR}" \
  --export=ALL,BASE_DIR="${BASE_DIR}",MANIFEST="${MANIFEST}" \
  "${DIR}/sbatch_watch_multi.sh")
watch_jid="${watch_jid%%;*}"
echo "[submit] watch jid=${watch_jid} eval_steps=${EVAL_STEPS}"

REPORT="${BASE_DIR}/log/train_code/ocr_extend300/submit_report.${jid_1p7b}_${jid_olmo}.txt"
{
  echo "submitted_at=$(date -Is)"
  echo "train_1p7b=${jid_1p7b}"
  echo "train_olmo=${jid_olmo}"
  echo "watch=${watch_jid}"
  echo "resume_1p7b=${CKPT_1P7B}"
  echo "resume_olmo=${CKPT_OLMO}"
  echo "max_steps=300 save_steps=25 eval_steps=${EVAL_STEPS}"
  echo "manifest=${MANIFEST}"
} | tee "${REPORT}"

echo "[done] report=${REPORT}"
cat "${MANIFEST}"
