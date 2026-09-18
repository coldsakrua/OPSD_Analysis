#!/bin/bash
# Submit Qwen3-1.7B / 4B / 8B snt_tt same-prefix (no privilege) trains,
# write manifest, then start a CPU-node watcher that submits ckpt-100 think evals.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/snt_tt_same_clip005/submit_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/snt_tt_same_clip005"
mkdir -p "${OUT_DIR}" log/train/{1.7b,4b,8b}
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

declare -a JOBS=(
  "qwen3_1.7b|qwen3_1.7b|snt_tt_same_clip005_1e_6_openthoughts_1p7b|scripts/train/qwen3_1.7b/withoutgt/snt_tt_same_clip005_ot_1e6.sh|snt_tt_same_clip005_1p7b_ckpt100"
  "qwen3_4b|qwen3_4b|snt_tt_same_clip005_1e_6_openthoughts_4b|scripts/train/qwen3_4b/withoutgt/snt_tt_same_clip005_ot_1e6.sh|snt_tt_same_clip005_4b_ckpt100"
  "qwen3_8b|qwen3_8b|snt_tt_same_clip005_1e_6_openthoughts_8b|scripts/train/qwen3_8b/withoutgt/snt_tt_same_clip005_ot_1e6.sh|snt_tt_same_clip005_8b_ckpt100"
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
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${ROOT}/scripts/train/snt_tt_same_clip005/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
