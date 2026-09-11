#!/bin/bash
# Submit Olmo-3-7B-Think OPSD train for jsd_token_clip=0.2,
# write manifest, then start a CPU-node watcher that submits ckpt-100 evals.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/olmo3_7b_think/jsd020/submit_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/olmo3_7b_think_jsd_clip"
mkdir -p "${OUT_DIR}" log/train/olmo3-7b-think
MANIFEST="${OUT_DIR}/submit_jsd020_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_jsd020_${STAMP}.tsv"

declare -a JOBS=(
  "olmo3_7b_think|olmo3_7b_think|st_tt_clip02_1e_6_openthoughts_olmo7bt|scripts/train/olmo3_7b_think/jsd020/opsd_student_think_teacher_think_clip02_1e_6_openthoughts.sh|st_tt_clip02_1e6_olmo7bt_ckpt100"
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
  echo "[submit] clip=${eval_tag} -> jid=${jid} script=${script}"
done

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"
echo "[submit] stamped=${MANIFEST_STAMPED}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_olmo_jsd020 \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${ROOT}/scripts/train/olmo3_7b_think/jsd_clip_sweep/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
