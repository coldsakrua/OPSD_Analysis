#!/bin/bash
# Submit two OPSD trains (1.7B-Base SFT@15000 → 1.7B think teacher,
# 4B-Base SFT@15000 → 4B think teacher), write manifest, then start a
# CPU compute-node watcher that submits ckpt-100 evals.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/sft15000_t_think/submit_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/sft15000_t_think"
mkdir -p "${OUT_DIR}" log/eval/qwen3_1.7b_base log/eval/qwen3_4b_base
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

declare -a JOBS=(
  "qwen3_1.7b_base|qwen3_1.7b_base|s1p7b_sft15000_t1p7b_think_clip005_1e_6_ot|scripts/train/sft15000_t_think/opsd_s1p7b_sft15000_t1p7b_think_clip005_1e6_ot.sh|sft15000_t1p7b_think_ckpt100"
  "qwen3_4b_base|qwen3_4b_base|s4b_sft15000_t4b_think_clip005_1e_6_ot|scripts/train/sft15000_t_think/opsd_s4b_sft15000_t4b_think_clip005_1e6_ot.sh|sft15000_t4b_think_ckpt100"
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
  jid="${jid%%;*}"
  echo -e "${model_key}\t${model_tag}\t${run_name}\t${jid}\t${script}\t${eval_tag}" >>"${MANIFEST_STAMPED}"
  echo "[submit] ${eval_tag} -> jid=${jid} script=${script}"
done

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"
echo "[submit] stamped=${MANIFEST_STAMPED}"

chmod +x "${ROOT}/scripts/train/sft15000_t_think/sbatch_watch.sh" \
  "${ROOT}/scripts/train/sft15000_t_think/watch_then_eval.sh" \
  "${ROOT}/scripts/train/sft15000_t_think/submit_four_think_evals.sh"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${ROOT}/scripts/train/sft15000_t_think/sbatch_watch.sh")
watch_jid="${watch_jid%%;*}"
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
