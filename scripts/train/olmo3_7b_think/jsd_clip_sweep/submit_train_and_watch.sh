#!/bin/bash
# Submit Olmo-3-7B-Think jsd clip=0.5 / 1.0 trains, write manifest,
# then start a CPU-node watcher that submits ckpt-100 think evals (4 datasets).
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/olmo3_7b_think/jsd_clip_sweep/submit_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/olmo3_7b_think_jsd_clip"
mkdir -p "${OUT_DIR}" log/train/olmo3-7b-think
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

declare -a JOBS=(
  "olmo3_7b_think|olmo3_7b_think|st_tt_clip05_1e_6_openthoughts_olmo7bt|scripts/train/olmo3_7b_think/jsd050/opsd_student_think_teacher_think_clip05_1e_6_openthoughts.sh|st_tt_clip05_olmo7bt_ckpt100"
  "olmo3_7b_think|olmo3_7b_think|st_tt_clip1_1e_6_openthoughts_olmo7bt|scripts/train/olmo3_7b_think/jsd100/opsd_student_think_teacher_think_clip1_1e_6_openthoughts.sh|st_tt_clip1_olmo7bt_ckpt100"
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

chmod +x \
  "${ROOT}/scripts/train/olmo3_7b_think/jsd_clip_sweep/sbatch_watch.sh" \
  "${ROOT}/scripts/train/olmo3_7b_think/jsd_clip_sweep/watch_then_eval.sh" \
  "${ROOT}/scripts/train/olmo3_7b_think/jsd_clip_sweep/submit_four_think_evals.sh"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_olmo_jsdclip \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${ROOT}/scripts/train/olmo3_7b_think/jsd_clip_sweep/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
