#!/bin/bash
# Submit Qwen3-1.7B think + Qwen3-4B-Instruct teacher OPSD (privileged + same-prompt),
# write manifest, then start a CPU compute-node watcher that submits ckpt-100 think evals.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/s1.7b_t4b_it/submit_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/s1.7b_t4b_it"
mkdir -p "${OUT_DIR}" log/eval/1.7b/{aime24,aime25,aime26,hmmt25}/think
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

declare -a JOBS=(
  "qwen3_1.7b|s1.7b_t4b_it|s1p7_t4bit_st_tnt_clip005_1e_6_openthoughts|scripts/train/s1.7b_t4b_it/jsd005/opsd_student_think_teacher_instruct_clip005_1e_6_openthoughts.sh|s1p7_t4bit_opsd_ckpt100"
  "qwen3_1.7b|s1.7b_t4b_it|s1p7_t4bit_st_tnt_same_clip005_1e_6_openthoughts|scripts/train/s1.7b_t4b_it/jsd005/same_student_think_teacher_instruct_clip005_1e_6_openthoughts.sh|s1p7_t4bit_same_ckpt100"
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

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_s1p7_t4bit \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${ROOT}/scripts/train/s1.7b_t4b_it/sbatch_watch.sh")
watch_jid="${watch_jid%%;*}"
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
