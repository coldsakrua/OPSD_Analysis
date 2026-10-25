#!/bin/bash
#SBATCH --job-name=watch_grpo_then_8b_opsd
#SBATCH --output=log/train/omr_answer_50step/watch_grpo_then_8b_opsd.%j.out
#SBATCH --partition=C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=5-00:00:00
set -euo pipefail

# Wait until GRPO train GRPO_JID enters RUNNING, then submit 8B OMR-answer OPSD.
# Required export: GRPO_JID (default 3890538)

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
cd "${BASE_DIR}"

GRPO_JID=${GRPO_JID:-3890538}
INTERVAL=${INTERVAL:-60}
SUBMIT_SCRIPT=${SUBMIT_SCRIPT:-scripts/train/omr_answer_50step/submit_train_and_watch_8b.sh}

log() { echo "[$(date -Is)] $*"; }

log "wait for GRPO_JID=${GRPO_JID} -> RUNNING, then run ${SUBMIT_SCRIPT}"

while true; do
  st=$(squeue -j "${GRPO_JID}" -h -o '%T' 2>/dev/null || true)
  if [[ -z "${st}" ]]; then
    # left queue; check final state
    final=$(sacct -j "${GRPO_JID}" -n -X -o State -P 2>/dev/null | head -1 | tr -d ' ' || true)
    log "GRPO ${GRPO_JID} left queue with state=${final:-UNKNOWN}; abort without submitting OPSD"
    exit 1
  fi
  log "GRPO ${GRPO_JID} state=${st}"
  if [[ "${st}" == "RUNNING" ]]; then
    break
  fi
  sleep "${INTERVAL}"
done

log "GRPO ${GRPO_JID} is RUNNING; submitting 8B OPSD"
bash "${SUBMIT_SCRIPT}"
log "done"
