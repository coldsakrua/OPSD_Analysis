#!/bin/bash
#SBATCH --job-name=watch_grpo_apps_eval
#SBATCH --output=log/train_code/sft_rl/watch_grpo_apps_then_merge_eval.%j.out
#SBATCH --partition=C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=5-00:00:00
set -euo pipefail

# Watch a GRPO APPS train. When global_step_${CKPT_STEP} exists (or train ends
# with that ckpt), merge FSDP actor → HF and submit 3 code think evals.
#
# Required env (or defaults for current 8B run):
#   TRAIN_JID MODEL_TAG RUN_NAME MODEL_KEY EVAL_TAG [CKPT_STEP=50]

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
cd "${BASE_DIR}"

TRAIN_JID=${TRAIN_JID:-4743996}
MODEL_TAG=${MODEL_TAG:-qwen3_8b}
RUN_NAME=${RUN_NAME:-grpo_think_8b_apps_4gpu}
MODEL_KEY=${MODEL_KEY:-qwen3_8b}
CKPT_STEP=${CKPT_STEP:-50}
EVAL_TAG=${EVAL_TAG:-grpo_think_8b_apps_4gpu_${TRAIN_JID}_gs${CKPT_STEP}}
INTERVAL=${INTERVAL:-180}
POST_DONE_WAIT=${POST_DONE_WAIT:-120}

ROOT_OUT="${BASE_DIR}/outputs/${MODEL_TAG}/${RUN_NAME}/${TRAIN_JID}"
ACTOR="${ROOT_OUT}/global_step_${CKPT_STEP}/actor"
MERGED="${ACTOR}/hf_merged"
STATE_DIR="${BASE_DIR}/log/train_code/ocr/watch_state"
STATE_FILE="${STATE_DIR}/${MODEL_KEY}_${EVAL_TAG}.done"
mkdir -p "${STATE_DIR}" log/train_code/sft_rl

log() { echo "[$(date -Is)] $*"; }

job_in_queue() {
  [[ -n "$(squeue -j "$1" -h -o '%T' 2>/dev/null || true)" ]]
}

job_state() {
  local st
  st=$(squeue -j "$1" -h -o '%T' 2>/dev/null || true)
  if [[ -n "${st}" ]]; then
    echo "${st}"
    return 0
  fi
  sacct -j "$1" -n -X -o State -P 2>/dev/null | head -1 | tr -d ' ' || echo UNKNOWN
}

hf_ok() {
  local p=$1
  [[ -f "${p}/config.json" ]] || return 1
  [[ -f "${p}/model.safetensors" ]] \
    || [[ -f "${p}/model.safetensors.index.json" ]] \
    || [[ -f "${p}/model-00001-of-00001.safetensors" ]] \
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

actor_ready() {
  [[ -d "${ACTOR}" ]] || return 1
  # FSDP shards present (at least rank0 model)
  [[ -f "${ACTOR}/model_world_size_4_rank_0.pt" ]] \
    || [[ -f "${ACTOR}/model_world_size_8_rank_0.pt" ]] \
    || compgen -G "${ACTOR}/model_world_size_*_rank_0.pt" >/dev/null
}

do_merge_and_eval() {
  if [[ -f "${STATE_FILE}" ]]; then
    log "already done: ${STATE_FILE}"
    return 0
  fi
  log "actor ready: ${ACTOR}"
  sleep "${POST_DONE_WAIT}"
  actor_ready || { log "actor disappeared after wait"; return 1; }

  if hf_ok "${MERGED}"; then
    log "reuse merged ${MERGED}"
  else
    log "merging FSDP -> HF"
    set +u
    # shellcheck disable=SC1091
    source activate anchor
    set -u
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export PYTHONPATH="${BASE_DIR}/vendor/verl:${BASE_DIR}/src:${PYTHONPATH:-}"
    mkdir -p "${MERGED}"
    python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "${ACTOR}" \
      --target_dir "${MERGED}" \
      --use_cpu_initialization
    hf_ok "${MERGED}" || { log "merge failed"; ls -lah "${MERGED}" || true; return 1; }
    log "merge done"
  fi

  rm -f "${BASE_DIR}/log/train_code/ocr/watch_state/${MODEL_KEY}_${EVAL_TAG}.submitted"
  MODEL_KEY="${MODEL_KEY}" CHECKPOINT_PATH="${MERGED}" EVAL_TAG="${EVAL_TAG}" \
    SEED=42 BASE_DIR="${BASE_DIR}" AUTO_JUDGE=1 \
    bash "${BASE_DIR}/scripts/train_code/ocr/submit_three_code_evals.sh"
  date -Is >"${STATE_FILE}"
  echo "${MERGED}" >>"${STATE_FILE}"
  log "submitted 3 code evals tag=${EVAL_TAG} ckpt=${MERGED}"
}

log "watch GRPO train_jid=${TRAIN_JID} run=${RUN_NAME} ckpt=global_step_${CKPT_STEP}"
log "expect actor=${ACTOR}"

while true; do
  if [[ -f "${STATE_FILE}" ]]; then
    log "state already done; exit"
    exit 0
  fi

  if actor_ready; then
    do_merge_and_eval
    exit 0
  fi

  if job_in_queue "${TRAIN_JID}"; then
    log "WAIT train state=$(job_state "${TRAIN_JID}") no global_step_${CKPT_STEP} yet"
    sleep "${INTERVAL}"
    continue
  fi

  st=$(job_state "${TRAIN_JID}")
  log "train left queue state=${st}"
  # give filesystem a moment
  sleep "${POST_DONE_WAIT}"
  if actor_ready; then
    do_merge_and_eval
    exit 0
  fi
  case "${st}" in
    COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|UNKNOWN)
      log "FAIL no usable global_step_${CKPT_STEP} under ${ROOT_OUT}"
      ls -la "${ROOT_OUT}" 2>/dev/null || true
      exit 1
      ;;
    *)
      sleep "${INTERVAL}"
      ;;
  esac
done
