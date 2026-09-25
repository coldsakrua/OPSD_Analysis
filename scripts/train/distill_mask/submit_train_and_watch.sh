#!/bin/bash
# Submit 4 distill-mask OPSD trains + CPU watcher that submits evals on ckpt-100.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/distill_mask/submit_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/distill_mask"
mkdir -p "${OUT_DIR}" log/train/{1.7b,4b,4b-instruct,olmo3-7b-instruct}
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

# model_key | model_tag | run_name | train_script | eval_tag (short, for state files)
declare -a JOBS=(
  "qwen3_1.7b|qwen3_1.7b|st_tt_clip005_1e_6_openthoughts_1p7b_mask_reasoning|scripts/train/qwen3_1.7b/jsd005/opsd_st_tt_clip005_1e_6_openthoughts_mask_reasoning.sh|maskR_1p7b_ckpt100"
  "qwen3_4b|qwen3_4b|st_tt_clip005_1e_6_openthoughts_4b_mask_reasoning|scripts/train/qwen3_4b/jsd005/opsd_st_tt_clip005_1e_6_openthoughts_mask_reasoning.sh|maskR_4b_ckpt100"
  "qwen3_4b_instruct|qwen3_4b_instruct|snt_tnt_clip005_1e_6_openthoughts_instruct_mask_structure|scripts/train/qwen3_4b_instruct/jsd005/opsd_snt_tnt_clip005_1e_6_openthoughts_mask_structure.sh|maskS_4bi_ckpt100"
  "olmo3_7b_instruct|olmo3_7b_instruct|snt_tnt_clip005_1e_6_openthoughts_olmo7bit_mask_structure|scripts/train/olmo3_7b_instruct/jsd005/opsd_snt_tnt_clip005_1e_6_openthoughts_mask_structure.sh|maskS_olmo7bi_ckpt100"
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
  echo "[submit] ${eval_tag} -> train_jid=${jid} script=${script}"
done

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_distill_mask \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${ROOT}/scripts/train/distill_mask/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
