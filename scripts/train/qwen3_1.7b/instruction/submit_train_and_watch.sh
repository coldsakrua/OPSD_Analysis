#!/bin/bash
# Submit Qwen3-1.7B st_tt instruction (no privilege, English concise vs detailed prefixes),
# write manifest, then start a CPU-node watcher that submits ckpt-100 think evals.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/qwen3_1.7b/instruction/submit_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/qwen3_1p7b_instr"
mkdir -p "${OUT_DIR}" log/train/1.7b
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

declare -a JOBS=(
  "qwen3_1.7b|qwen3_1.7b|st_tt_instr_clip005_1e_6_openthoughts_1p7b|scripts/train/qwen3_1.7b/instruction/opsd_st_tt_instr_clip005_1e6_ot.sh|st_tt_instr_clip005_1p7b_ckpt100"
)

{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript\teval_tag"
} >"${MANIFEST_STAMPED}"

echo "[submit] root=${ROOT}"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r model_key model_tag run_name script eval_tag <<<"${entry}"
  if [[ ! -f "${script}" ]]; then
    echo "[error] missing ${script}" >&2
    exit 1
  fi
  chmod +x "${script}"
  jid=$(sbatch --parsable --chdir="${ROOT}" "${script}")
  echo -e "${model_key}\t${model_tag}\t${run_name}\t${jid}\t${script}\t${eval_tag}" >>"${MANIFEST_STAMPED}"
  echo "[submit] ${eval_tag} -> jid=${jid} script=${script}"
done

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"
echo "[submit] stamped=${MANIFEST_STAMPED}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_1p7b_instr \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${ROOT}/scripts/train/qwen3_1.7b/instruction/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
