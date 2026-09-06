#!/bin/bash
# Submit all robustness trains (topk + multi-seed), write manifest, start CPU watcher.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/robustness"
mkdir -p "${OUT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"
MANIFEST="${OUT_DIR}/submit_latest.tsv"
FAIL_LOG="${OUT_DIR}/submit_failures_${STAMP}.txt"
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
  # sbatch --parsable may return jobid;cluster
  jid="${jid%%;*}"
  echo -e "${model_key}\t${model_tag}\t${run_name}\t${jid}\t${script}" >>"${MANIFEST_STAMPED}"
  echo "[submit] ${model_key} ${run_name} -> jid=${jid}"
  return 0
}

echo "[submit] root=${ROOT} stamp=${STAMP}"

# ----- Top-k KL -----
for k in 1 4 16; do
  submit_one "qwen3_1.7b" "qwen3_1.7b" "st_tt_clip005_topk${k}_c1024_1p7b" \
    "scripts/train/qwen3_1.7b/topk/opsd_st_tt_clip005_topk${k}_c1024_ot.sh" || true
done
for k in 1 4 16; do
  submit_one "olmo3_7b_think" "olmo3_7b_think" "st_tt_clip005_topk${k}_rkl_c1024_olmo7bt" \
    "scripts/train/olmo3_7b_think/topk/opsd_st_tt_clip005_topk${k}_c1024_ot.sh" || true
done
for k in 1 4 16; do
  submit_one "qwen3_4b_instruct" "qwen3_4b_instruct" "snt_tnt_clip005_topk${k}_c1024_oti" \
    "scripts/train/qwen3_4b_instruct/topk/opsd_snt_tnt_clip005_topk${k}_c1024_ot.sh" || true
done

# ----- Multi-seed trains -----
VARIANTS=(c256 c1024 answer ios first256 uni256 last256)
for seed in 1024 65536; do
  for name in "${VARIANTS[@]}"; do
    submit_one "qwen3_1.7b" "qwen3_1.7b" "st_tt_clip005_${name}_seed${seed}_1p7b" \
      "scripts/train/qwen3_1.7b/jsd005/seed/opsd_st_tt_clip005_${name}_seed${seed}_ot.sh" || true
  done
  for name in "${VARIANTS[@]}"; do
    submit_one "olmo3_7b_think" "olmo3_7b_think" "st_tt_clip005_${name}_seed${seed}_olmo7bt" \
      "scripts/train/olmo3_7b_think/jsd005/seed/opsd_st_tt_clip005_${name}_seed${seed}_ot.sh" || true
  done
done

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
n_ok=$(grep -cvE '^\s*(#|$)' "${MANIFEST}" || true)
echo "[submit] manifest=${MANIFEST} ok=${n_ok}"
echo "[submit] stamped=${MANIFEST_STAMPED}"
if [[ -s "${FAIL_LOG}" ]]; then
  echo "[submit] failures logged in ${FAIL_LOG}"
  cat "${FAIL_LOG}"
fi

if [[ "${n_ok}" -eq 0 ]]; then
  echo "[error] no trains submitted; skip watch" >&2
  exit 1
fi

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${ROOT}/scripts/train/robustness/sbatch_watch.sh")
watch_jid="${watch_jid%%;*}"
echo "[submit] watch -> jid=${watch_jid}"
echo "${watch_jid}" >"${OUT_DIR}/watch_jid_${STAMP}.txt"
echo "[submit] done"
cat "${MANIFEST}"
