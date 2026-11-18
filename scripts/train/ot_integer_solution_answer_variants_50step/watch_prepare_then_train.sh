#!/bin/bash
#SBATCH --job-name=watch_prep_ot_int_sol_ans_var
#SBATCH --output=log/train/ot_integer_solution_answer_variants_50step/watch_prep.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodelist=a6n18c1,a6n18c2,a6n18c3,a6n18c4,a6n17c1,a6n17c2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:00:00
set -euo pipefail

# Wait for prepare job to finish, then submit 8 trains + eval watcher.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
cd "${BASE_DIR}"
: "${PREP_JID:?Set PREP_JID}"

log() { echo "[$(date -Is)] $*"; }

log "waiting for prepare jid=${PREP_JID} host=$(hostname)"
while true; do
  st=$(squeue -j "${PREP_JID}" -h -o '%T' 2>/dev/null || true)
  if [[ -n "${st}" ]]; then
    log "prepare still ${st}"
    sleep 30
    continue
  fi
  break
done

acct=$(sacct -j "${PREP_JID}" -n -X -o State -P 2>/dev/null | head -1 | tr -d ' ' || echo UNKNOWN)
log "prepare left queue state=${acct}"
case "${acct}" in
  COMPLETED|COMPLETED+)
    ;;
  *)
    log "prepare not COMPLETED (${acct}); abort train submit"
    exit 1
    ;;
esac

PRE="${BASE_DIR}/data/openthoughts/preprocessed"
for ds in \
  "${PRE}/openthoughts.opsd.solution.sthink_tthink.integer_answer.rewritten_frac_decimal.sampler_seed1.n3200.parquet" \
  "${PRE}/openthoughts.opsd.solution.sthink_tthink.integer_answer.wrong_n10p1.sampler_seed1.n3200.parquet" \
  "${PRE}/openthoughts.opsd.solution.sthink_tthink.integer_answer.wrong_rand50.sampler_seed1.n3200.parquet"
do
  if [[ ! -f "${ds}" ]]; then
    log "missing ${ds}"
    exit 1
  fi
done

log "prepare OK; submitting trains + watch"
bash "${BASE_DIR}/scripts/train/ot_integer_solution_answer_variants_50step/submit_train_and_watch.sh"
log "done"
