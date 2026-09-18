#!/bin/bash
#SBATCH --job-name=watch_grpo4b_gs50
#SBATCH --output=log/train/rl/qwen3_4b_think/watch_gs50.%j.out
#SBATCH --partition=C64M256G,C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
set -euo pipefail

# Wait until WAIT_FOR_JID is RUNNING, then submit 4 think evals for
# GRPO 4B global_step_50 (merge must already exist).

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
WAIT_FOR_JID=${WAIT_FOR_JID:-3749337}
INTERVAL=${INTERVAL:-60}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${BASE_DIR}/outputs/qwen3_4b/grpo_think_4b_omr_int_8gpu_bs64_24k_6400/3719725/global_step_50/actor/hf_merged}
EVAL_TAG=${EVAL_TAG:-grpo_omr_int_4b_8gpu_bs64_24k_gs50}
STATE_DIR="${BASE_DIR}/log/train/rl/qwen3_4b_think/watch_state"
STAMP="${STATE_DIR}/qwen3_4b_3719725_gs50.submitted"
ARRAY_WORKER="${BASE_DIR}/scripts/eval/common/sbatch_four_array.sh"
ARRAY_SCRIPT_DIR="${BASE_DIR}/scripts/eval/4b"
LOG="${BASE_DIR}/log/train/rl/qwen3_4b_think/watch_gs50.${SLURM_JOB_ID:-manual}.log"

cd "${BASE_DIR}"
mkdir -p "${STATE_DIR}" log/eval/4b/array "$(dirname "${LOG}")"

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

job_state() {
  local jid=$1 st
  st=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
  if [[ -n "${st}" ]]; then
    echo "${st}"
    return 0
  fi
  st=$(sacct -j "${jid}" -n -X -o State -P 2>/dev/null | head -1 | tr -d ' ' || true)
  echo "${st:-UNKNOWN}"
}

normalize_state() {
  local st=$1
  echo "${st%%+*}"
}

hf_merged_ready() {
  local p=$1
  [[ -f "${p}/config.json" ]] || return 1
  [[ -f "${p}/model.safetensors" ]] \
    || [[ -f "${p}/model.safetensors.index.json" ]] \
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

log "host=$(hostname) job=${SLURM_JOB_ID:-none} wait_for=${WAIT_FOR_JID} interval=${INTERVAL}s"
log "ckpt=${CHECKPOINT_PATH} tag=${EVAL_TAG}"

if [[ -f "${STAMP}" ]]; then
  log "SKIP already submitted: ${STAMP}"
  cat "${STAMP}" | tee -a "${LOG}"
  exit 0
fi

if ! hf_merged_ready "${CHECKPOINT_PATH}"; then
  log "ERROR incomplete merge: ${CHECKPOINT_PATH}"
  ls -lah "${CHECKPOINT_PATH}" >>"${LOG}" 2>&1 || true
  exit 1
fi

while true; do
  st=$(normalize_state "$(job_state "${WAIT_FOR_JID}")")
  case "${st}" in
    RUNNING|COMPLETING)
      log "WAIT_FOR ${WAIT_FOR_JID} state=${st} — submit evals"
      break
      ;;
    COMPLETED)
      log "WAIT_FOR ${WAIT_FOR_JID} already COMPLETED — submit evals"
      break
      ;;
    FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
      log "WAIT_FOR ${WAIT_FOR_JID} ended ${st} — submit evals anyway"
      break
      ;;
    PENDING|CONFIGURING|REQUEUED|RESIZING|SUSPENDED)
      log "HOLD until ${WAIT_FOR_JID} is RUNNING (now ${st})"
      sleep "${INTERVAL}"
      ;;
    *)
      log "HOLD ${WAIT_FOR_JID} state=${st}"
      sleep "${INTERVAL}"
      ;;
  esac
done

log "SUBMIT evals ckpt=${CHECKPOINT_PATH} tag=${EVAL_TAG}"
eval_jid=$(sbatch --parsable --chdir="${BASE_DIR}" \
  --array=0-3 \
  --job-name="grpo4b_gs50" \
  --output="${BASE_DIR}/log/eval/4b/array/%x_%A_%a.out" \
  --time=24:00:00 \
  --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",EVAL_TAG="${EVAL_TAG}",SEED=42,ARRAY_SCRIPT_DIR="${ARRAY_SCRIPT_DIR}",ARRAY_SCRIPT_SUFFIX="_think",ARRAY_OUTPUT_JSON_FMT="${BASE_DIR}/eval_outputs/${EVAL_TAG}/__DS___4b_think.json" \
  "${ARRAY_WORKER}")
eval_jid="${eval_jid%%;*}"
log "SUBMITTED eval_array=${eval_jid} tasks=0-3"

{
  date -Is
  echo "merged=${CHECKPOINT_PATH}"
  echo "eval_tag=${EVAL_TAG}"
  echo "wait_for_jid=${WAIT_FOR_JID}"
  echo "eval_array=${eval_jid}"
} >"${STAMP}"
log "stamp=${STAMP}"
log "watch_gs50 finished"
exit 0
