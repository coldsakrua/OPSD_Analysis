#!/bin/bash
#SBATCH --job-name=watch_ot_int_sol_ans_var_50step
#SBATCH --output=log/train/ot_integer_solution_answer_variants_50step/watch_slurm.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodelist=a6n18c1,a6n18c2,a6n18c3,a6n18c4,a6n17c1,a6n17c2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
set -euo pipefail

# Watch 8 OT-integer **solution** answer-variant 50-step trains.
# On complete checkpoint-50, submit 4 think evals per run.
# Required via sbatch --export: TRAIN_JID_* for all 8 runs.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
cd "${BASE_DIR}"

: "${TRAIN_JID_17S:?}"
: "${TRAIN_JID_17R:?}"
: "${TRAIN_JID_17W:?}"
: "${TRAIN_JID_17X:?}"
: "${TRAIN_JID_OLS:?}"
: "${TRAIN_JID_OLR:?}"
: "${TRAIN_JID_OLW:?}"
: "${TRAIN_JID_OLX:?}"

RUN_NAME_17S=${RUN_NAME_17S:-st_tt_clip005_1e_6_ot_integer_solution_50step_1p7b}
RUN_NAME_17R=${RUN_NAME_17R:-st_tt_clip005_1e_6_ot_integer_solution_rewritten_50step_1p7b}
RUN_NAME_17W=${RUN_NAME_17W:-st_tt_clip005_1e_6_ot_integer_solution_wrong_50step_1p7b}
RUN_NAME_17X=${RUN_NAME_17X:-st_tt_clip005_1e_6_ot_integer_solution_wrong_rand50_50step_1p7b}
RUN_NAME_OLS=${RUN_NAME_OLS:-st_tt_clip005_1e_6_ot_integer_solution_50step_olmo7bt}
RUN_NAME_OLR=${RUN_NAME_OLR:-st_tt_clip005_1e_6_ot_integer_solution_rewritten_50step_olmo7bt}
RUN_NAME_OLW=${RUN_NAME_OLW:-st_tt_clip005_1e_6_ot_integer_solution_wrong_50step_olmo7bt}
RUN_NAME_OLX=${RUN_NAME_OLX:-st_tt_clip005_1e_6_ot_integer_solution_wrong_rand50_50step_olmo7bt}
CKPT_STEP=${CKPT_STEP:-50}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}
EVAL_SEED=${EVAL_SEED:-42}

STATE_DIR="${BASE_DIR}/log/train/ot_integer_solution_answer_variants_50step/watch_state"
mkdir -p "${STATE_DIR}" "log/train/ot_integer_solution_answer_variants_50step"

log() { echo "[$(date -Is)] $*"; }

job_in_queue() {
  local jid=$1
  [[ -n "$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)" ]]
}

job_state() {
  local jid=$1 st
  st=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
  if [[ -n "${st}" ]]; then
    echo "${st}"
    return 0
  fi
  sacct -j "${jid}" -n -X -o State -P 2>/dev/null | head -1 | tr -d ' ' || echo UNKNOWN
}

ckpt_has_weights() {
  local p=$1
  [[ -f "${p}/config.json" ]] || return 1
  [[ -f "${p}/model.safetensors" ]] \
    || [[ -f "${p}/model.safetensors.index.json" ]] \
    || [[ -f "${p}/model-00001-of-00001.safetensors" ]] \
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

ckpt_complete() {
  local p=$1
  ckpt_has_weights "${p}" || return 1
  [[ -f "${p}/trainer_state.json" ]] || return 1
}

find_ckpt() {
  local model_tag=$1 run_name=$2 train_jid=$3
  local root="${BASE_DIR}/outputs/${model_tag}/${run_name}/${train_jid}"
  if ckpt_complete "${root}/checkpoint-${CKPT_STEP}"; then
    echo "${root}/checkpoint-${CKPT_STEP}"
    return 0
  fi
  if ckpt_complete "${root}/final"; then
    echo "${root}/final"
    return 0
  fi
  return 1
}

submit_evals() {
  local eval_family=$1 ckpt=$2 eval_tag=$3
  case "${eval_family}" in
    1.7b)
      CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${eval_tag}" SEED="${EVAL_SEED}" BASE_DIR="${BASE_DIR}" \
        bash "${BASE_DIR}/scripts/eval/1.7b/submit_four.sh"
      ;;
    olmo_7b_think)
      CHECKPOINT_PATH="${ckpt}" EVAL_TAG="${eval_tag}" SEED="${EVAL_SEED}" BASE_DIR="${BASE_DIR}" \
        bash "${BASE_DIR}/scripts/eval/olmo_7b_think/seed/submit_four_seed.sh"
      ;;
    *)
      echo "[error] unknown eval_family=${eval_family}" >&2
      return 1
      ;;
  esac
}

handle_one() {
  local key=$1 model_tag=$2 run_name=$3 train_jid=$4 eval_family=$5
  local eval_tag="${run_name}_ckpt${CKPT_STEP}"
  local stamp="${STATE_DIR}/${key}_${train_jid}.done"
  local fail_stamp="${STATE_DIR}/${key}_${train_jid}.failed"

  if [[ -f "${stamp}" ]]; then
    return 0
  fi

  local ckpt=""
  if ckpt=$(find_ckpt "${model_tag}" "${run_name}" "${train_jid}"); then
    log "READY ${key} jid=${train_jid} -> ${ckpt} (train_state=$(job_state "${train_jid}"))"
    sleep "${POST_DONE_WAIT}"
    if ! ckpt=$(find_ckpt "${model_tag}" "${run_name}" "${train_jid}"); then
      log "WAIT ${key} jid=${train_jid} (ckpt disappeared after wait)"
      return 1
    fi
    if [[ "$(basename "${ckpt}")" != "checkpoint-${CKPT_STEP}" ]]; then
      eval_tag="${run_name}_final"
    fi
    if submit_evals "${eval_family}" "${ckpt}" "${eval_tag}"; then
      date -Is >"${stamp}"
      echo "${ckpt}" >>"${stamp}"
      rm -f "${fail_stamp}"
      log "SUBMITTED 4 evals for ${key} tag=${eval_tag} seed=${EVAL_SEED}"
      return 0
    fi
    log "ERROR submitting evals for ${key} (will retry)"
    return 1
  fi

  if job_in_queue "${train_jid}"; then
    log "WAIT ${key} jid=${train_jid} state=$(job_state "${train_jid}")"
    return 1
  fi

  local st
  st=$(job_state "${train_jid}")
  case "${st}" in
    FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED)
      log "FAIL ${key} jid=${train_jid} state=${st}, no usable checkpoint-${CKPT_STEP}/final"
      date -Is >"${fail_stamp}"
      echo "${st}" >>"${fail_stamp}"
      return 0
      ;;
    *)
      log "WAIT ${key} jid=${train_jid} left queue (state=${st}) but no usable ckpt yet"
      return 1
      ;;
  esac
}

log "host=$(hostname) job=${SLURM_JOB_ID:-manual} interval=${INTERVAL}s ckpt_step=${CKPT_STEP} eval_seed=${EVAL_SEED}"
log "watching 8 trains (solution / rewritten / wrong_n10p1 / wrong_rand50 × 1.7b+olmo)"

while true; do
  done_count=0
  handle_one "1p7b_sol" "qwen3_1.7b" "${RUN_NAME_17S}" "${TRAIN_JID_17S}" "1.7b" && done_count=$((done_count + 1))
  handle_one "1p7b_sol_rw" "qwen3_1.7b" "${RUN_NAME_17R}" "${TRAIN_JID_17R}" "1.7b" && done_count=$((done_count + 1))
  handle_one "1p7b_sol_wrong" "qwen3_1.7b" "${RUN_NAME_17W}" "${TRAIN_JID_17W}" "1.7b" && done_count=$((done_count + 1))
  handle_one "1p7b_sol_wr50" "qwen3_1.7b" "${RUN_NAME_17X}" "${TRAIN_JID_17X}" "1.7b" && done_count=$((done_count + 1))
  handle_one "olmo_sol" "olmo3_7b_think" "${RUN_NAME_OLS}" "${TRAIN_JID_OLS}" "olmo_7b_think" && done_count=$((done_count + 1))
  handle_one "olmo_sol_rw" "olmo3_7b_think" "${RUN_NAME_OLR}" "${TRAIN_JID_OLR}" "olmo_7b_think" && done_count=$((done_count + 1))
  handle_one "olmo_sol_wrong" "olmo3_7b_think" "${RUN_NAME_OLW}" "${TRAIN_JID_OLW}" "olmo_7b_think" && done_count=$((done_count + 1))
  handle_one "olmo_sol_wr50" "olmo3_7b_think" "${RUN_NAME_OLX}" "${TRAIN_JID_OLX}" "olmo_7b_think" && done_count=$((done_count + 1))
  log "progress done=${done_count}/8"
  if [[ "${done_count}" -ge 8 ]]; then
    log "ALL trains handled; exiting"
    break
  fi
  sleep "${INTERVAL}"
done

log "watch finished"
exit 0
