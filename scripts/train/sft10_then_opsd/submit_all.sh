#!/bin/bash
# Submit SFT10(OMR long CoT)→OPSD(same, OT1024) for 1.7b / olmo7bt / 4bt.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/sft10_then_opsd/submit_all.sh
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
cd "${BASE_DIR}"
DIR="${BASE_DIR}/scripts/train/sft10_then_opsd"

mkdir -p \
  log/train/sft10_then_opsd \
  log/train/1.7b \
  log/train/olmo3-7b-think \
  log/train/4b_thinking \
  log/eval/1.7b/array \
  log/eval/olmo_7b_think/array \
  log/eval/qwen3_4b_thinking

chmod +x \
  "${DIR}/sft10_opsd_qwen3_1p7b.sh" \
  "${DIR}/sft10_opsd_olmo7bt.sh" \
  "${DIR}/sft10_opsd_qwen3_4bt.sh" \
  "${DIR}/submit_four_think_evals.sh" \
  "${BASE_DIR}/scripts/eval/qwen3_4b_thinking/submit_four.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="${BASE_DIR}/log/train/sft10_then_opsd/submit_latest.tsv"
MANIFEST_STAMPED="${BASE_DIR}/log/train/sft10_then_opsd/submit_${STAMP}.tsv"

{
  echo -e "# model_key\tmodel_tag\tsft_run_name\topsd_run_name\ttrain_jid\tsft_step\topsd_step"
} >"${MANIFEST_STAMPED}"

echo "=== submit SFT10(OMR12k) → OPSD(same / OT1024) trains ==="

jid_1p7b=$(sbatch --parsable --chdir="${BASE_DIR}" "${DIR}/sft10_opsd_qwen3_1p7b.sh")
jid_1p7b="${jid_1p7b%%;*}"
echo -e "qwen3_1.7b\tqwen3_1.7b\tsft_think_10step_omr12k_1p7b\tst_tt_same_clip005_1e_6_ot1024_1p7b_sft10omr\t${jid_1p7b}\t10\t100" \
  >>"${MANIFEST_STAMPED}"
echo "[submit] 1.7b jid=${jid_1p7b}"

jid_olmo=$(sbatch --parsable --chdir="${BASE_DIR}" "${DIR}/sft10_opsd_olmo7bt.sh")
jid_olmo="${jid_olmo%%;*}"
echo -e "olmo3_7b_think\tolmo3_7b_think\tsft_think_10step_omr12k_olmo7bt\tst_tt_same_clip005_1e_6_ot1024_olmo7bt_sft10omr\t${jid_olmo}\t10\t100" \
  >>"${MANIFEST_STAMPED}"
echo "[submit] olmo7bt jid=${jid_olmo}"

jid_4bt=$(sbatch --parsable --chdir="${BASE_DIR}" "${DIR}/sft10_opsd_qwen3_4bt.sh")
jid_4bt="${jid_4bt%%;*}"
echo -e "qwen3_4b_thinking\tqwen3_4b_thinking\tsft_think_10step_omr12k_4bt\tst_tt_same_clip005_1e_6_ot1024_4bt_sft10omr\t${jid_4bt}\t10\t100" \
  >>"${MANIFEST_STAMPED}"
echo "[submit] 4bt jid=${jid_4bt}"

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"

REPORT="${BASE_DIR}/log/train/sft10_then_opsd/submit_report.${STAMP}.txt"
{
  echo "submitted_at=$(date -Is)"
  echo "sft_data=omr.integer_answer.think.le12k"
  echo "opsd_data=openthoughts maxprompt1024"
  echo "mode=same teacher_privilege_field=none"
  echo "train_1p7b=${jid_1p7b}"
  echo "train_olmo=${jid_olmo}"
  echo "train_4bt=${jid_4bt}"
  echo "manifest=${MANIFEST}"
  echo "stamped=${MANIFEST_STAMPED}"
} | tee "${REPORT}"

echo "[done] report=${REPORT}"
cat "${MANIFEST}"
