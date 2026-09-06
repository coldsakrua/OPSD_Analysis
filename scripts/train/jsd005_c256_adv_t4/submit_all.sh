#!/bin/bash
# Submit 5 c256+adv_t4 trains, write manifest, then start CPU watcher for think evals.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/jsd005_c256_adv_t4"
mkdir -p "${OUT_DIR}"
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

declare -a JOBS=(
  "qwen3_1.7b|qwen3_1.7b|st_tt_clip005_c256_adv_t4_1p7b|scripts/train/qwen3_1.7b/jsd005/opsd_st_tt_clip005_c256_adv_t4_openthoughts.sh"
  "qwen3_4b|qwen3_4b|st_tt_clip005_c256_adv_t4_4b|scripts/train/qwen3_4b/jsd005/opsd_st_tt_clip005_c256_adv_t4_openthoughts.sh"
  "qwen3_4b_thinking|qwen3_4b_thinking|st_tt_clip005_c256_adv_t4_4bt|scripts/train/qwen3_4b_thinking/jsd005/opsd_st_tt_clip005_c256_adv_t4_openthoughts.sh"
  "olmo3_7b_think|olmo3_7b_think|st_tt_clip005_c256_adv_t4_olmo7bt|scripts/train/olmo3_7b_think/jsd005/opsd_st_tt_clip005_c256_adv_t4_openthoughts.sh"
  "mimo_7b_rl|mimo_7b_rl|st_tt_clip005_c256_adv_t4_mimo7b|scripts/train/mimo_7b_rl/jsd005/opsd_st_tt_clip005_c256_adv_t4_openthoughts.sh"
)

{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript"
} >"${MANIFEST_STAMPED}"

echo "[submit] root=${ROOT}"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r model_key model_tag run_name script <<<"${entry}"
  if [[ ! -f "${script}" ]]; then
    echo "[error] missing ${script}" >&2
    exit 1
  fi
  jid=$(sbatch --parsable --chdir="${ROOT}" "${script}")
  echo -e "${model_key}\t${model_tag}\t${run_name}\t${jid}\t${script}" >>"${MANIFEST_STAMPED}"
  echo "[submit] ${model_key} -> jid=${jid} script=${script}"
done

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"
echo "[submit] stamped=${MANIFEST_STAMPED}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${ROOT}/scripts/train/jsd005_c256_adv_t4/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
