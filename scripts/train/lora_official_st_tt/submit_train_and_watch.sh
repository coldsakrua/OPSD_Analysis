#!/bin/bash
# Submit official-OPSD LoRA trains (4B-Thinking + Olmo-3-7B-Think), write manifest,
# then start a CPU watcher that submits 4 think evals when checkpoint-100 is saved.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/lora_official_st_tt/submit_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/lora_official_st_tt"
mkdir -p "${OUT_DIR}"
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

QWEN_SH="${ROOT}/scripts/train/qwen3_4b_thinking/lora/openthoughts/opsd_student_think_teacher_think_lora_openthoughts.sh"
OLMO_SH="${ROOT}/scripts/train/olmo3_7b_think/lora/openthoughts/opsd_student_think_teacher_think_lora_openthoughts.sh"

for f in "${QWEN_SH}" "${OLMO_SH}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[error] missing ${f}" >&2
    exit 1
  fi
  chmod +x "${f}"
done

{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript"
} >"${MANIFEST_STAMPED}"

echo "[submit] root=${ROOT}"

qwen_jid=$(sbatch --parsable --chdir="${ROOT}" "${QWEN_SH}")
echo -e "qwen3_4b_thinking\tqwen3_4b_thinking\tst_tt_lora_clip005_lr5e6_openthoughts_4bt\t${qwen_jid}\t${QWEN_SH}" >>"${MANIFEST_STAMPED}"
echo "[submit] qwen3_4b_thinking -> jid=${qwen_jid}"

olmo_jid=$(sbatch --parsable --chdir="${ROOT}" "${OLMO_SH}")
echo -e "olmo3_7b_think\tolmo3_7b_think\tst_tt_lora_clip005_lr5e6_openthoughts_olmo7bt\t${olmo_jid}\t${OLMO_SH}" >>"${MANIFEST_STAMPED}"
echo "[submit] olmo3_7b_think -> jid=${olmo_jid}"

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"
echo "[submit] stamped=${MANIFEST_STAMPED}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${ROOT}/scripts/train/lora_official_st_tt/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
squeue -u "${USER}" -o '%.18i %.12P %.22j %.8u %.2t %.10M %.4D %R' | head -30
