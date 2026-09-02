#!/bin/bash
# Submit c1024+uni256 and c1024+last256 train jobs, write manifest, then
# sbatch the CPU-node watcher that submits 4 think evals per finished run.
set -euo pipefail

BASE=/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis
cd "${BASE}"
mkdir -p log/train

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="${BASE}/log/train/c1024_uni_last256_submit_${STAMP}.tsv"
LATEST="${BASE}/log/train/c1024_uni_last256_submit_latest.tsv"

UNI_SCRIPT=scripts/train/qwen3_1.7b/jsd005/opsd_st_tt_clip005_c1024_uni256_openthoughts.sh
LAST_SCRIPT=scripts/train/qwen3_1.7b/jsd005/opsd_st_tt_clip005_c1024_last256_openthoughts.sh
WATCH_SCRIPT=scripts/train/qwen3_1.7b/jsd005/sbatch_watch_c1024_uni_last256.sh

for s in "${UNI_SCRIPT}" "${LAST_SCRIPT}" "${WATCH_SCRIPT}"; do
  if [[ ! -f "${s}" ]]; then
    echo "[error] missing ${s}" >&2
    exit 1
  fi
done

echo "[submit] base=${BASE}"
UNI_JID=$(sbatch --parsable "${UNI_SCRIPT}")
echo "[submit] uni256 train -> ${UNI_JID}"
LAST_JID=$(sbatch --parsable "${LAST_SCRIPT}")
echo "[submit] last256 train -> ${LAST_JID}"

{
  echo -e "job_id\tmodel_tag\tloss\trun_name\tscript"
  echo -e "${UNI_JID}\tqwen3_1.7b\tuni256\tst_tt_clip005_c1024_uni256_1p7b\t${UNI_SCRIPT}"
  echo -e "${LAST_JID}\tqwen3_1.7b\tlast256\tst_tt_clip005_c1024_last256_1p7b\t${LAST_SCRIPT}"
} > "${MANIFEST}"
ln -sfn "$(basename "${MANIFEST}")" "${LATEST}"
echo "[submit] manifest=${MANIFEST}"
echo "[submit] latest=${LATEST}"
cat "${MANIFEST}"

WATCH_JID=$(
  sbatch --parsable \
    --export=ALL,BASE_DIR="${BASE}",MANIFEST="${LATEST}",INTERVAL=120,POST_DONE_WAIT=60 \
    "${WATCH_SCRIPT}"
)
echo "[submit] watch (CPU) -> ${WATCH_JID}"
echo "[submit] watch log will be log/train/watch_c1024_uni_last256.${WATCH_JID}.out"
squeue -u "${USER}" -o '%.10i %.12P %.32j %.2t %.10M %R' | head -20
