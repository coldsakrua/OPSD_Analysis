#!/bin/bash
# Fill missing Olmo-3-7B-Think seed=42 uni256 / last256 trains, then CPU-watch
# for 4 evals (AIME24/25/26 + HMMT25, eval SEED=42) once checkpoint-100 is ready.
#
# Does NOT touch already-running/pending jobs:
#   last256 seed1024 train 3561990 (watched by 3561991)
#   last256 seed65536 evals 3563232
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/robustness"
mkdir -p "${OUT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_fill_olmo_s42_${STAMP}.tsv"
MANIFEST="${OUT_DIR}/submit_fill_olmo_s42_latest.tsv"
FAIL_LOG="${OUT_DIR}/submit_fill_olmo_s42_${STAMP}.fail.txt"
: >"${FAIL_LOG}"

{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript"
} >"${MANIFEST_STAMPED}"

submit_one() {
  local model_key=$1 model_tag=$2 run_name=$3 script=$4
  if [[ ! -f "${script}" ]]; then
    echo "[error] missing ${script}" | tee -a "${FAIL_LOG}" >&2
    return 1
  fi
  local jid
  if ! jid=$(sbatch --parsable --chdir="${ROOT}" "${script}" 2>>"${FAIL_LOG}"); then
    echo "[error] sbatch failed: ${script}" | tee -a "${FAIL_LOG}" >&2
    return 1
  fi
  jid="${jid%%;*}"
  echo -e "${model_key}\t${model_tag}\t${run_name}\t${jid}\t${script}" >>"${MANIFEST_STAMPED}"
  echo "[submit] ${run_name} -> jid=${jid}"
}

echo "[submit] root=${ROOT} stamp=${STAMP}"

submit_one "olmo3_7b_think" "olmo3_7b_think" "st_tt_clip005_c1024_uni256_olmo7bt" \
  "scripts/train/olmo3_7b_think/jsd005/opsd_st_tt_clip005_c1024_uni256_openthoughts.sh" || true
submit_one "olmo3_7b_think" "olmo3_7b_think" "st_tt_clip005_c1024_last256_olmo7bt" \
  "scripts/train/olmo3_7b_think/jsd005/opsd_st_tt_clip005_c1024_last256_openthoughts.sh" || true

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
n_ok=$(grep -cvE '^\s*(#|$)' "${MANIFEST}" || true)
echo "[submit] manifest=${MANIFEST} ok=${n_ok}"
if [[ -s "${FAIL_LOG}" ]]; then
  echo "[submit] failures logged in ${FAIL_LOG}"
  cat "${FAIL_LOG}"
fi
if [[ "${n_ok}" -eq 0 ]]; then
  echo "[error] no trains submitted; skip watch" >&2
  exit 1
fi

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=fill_olmo_s42_watch \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}",INTERVAL=120,POST_DONE_WAIT=60 \
  "${ROOT}/scripts/train/robustness/sbatch_watch.sh")
watch_jid="${watch_jid%%;*}"
echo "[submit] watch -> jid=${watch_jid}"
echo "${watch_jid}" >"${OUT_DIR}/watch_jid_fill_olmo_s42_${STAMP}.txt"
echo "[submit] done"
cat "${MANIFEST}"
