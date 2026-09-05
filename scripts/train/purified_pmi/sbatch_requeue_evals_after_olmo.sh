#!/bin/bash
#SBATCH --job-name=pmi_requeue_eval
#SBATCH --output=log/train/purified_pmi/%x.%j.out
#SBATCH --partition=C64M256G,C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-00:00:00
set -euo pipefail

# Wait until Olmo PMI trains are running (or finished), then resubmit
# cancelled qwen3_4b / qwen3_4b_thinking evals (4×4=16) and restart watch.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
cd "${BASE_DIR}"
SUBMIT="${BASE_DIR}/scripts/train/purified_pmi/submit_four_evals.sh"
WATCH="${BASE_DIR}/scripts/train/purified_pmi/sbatch_watch_eval.sh"
INTERVAL=${INTERVAL:-60}
chmod +x "${SUBMIT}" "${WATCH}" "${BASE_DIR}/scripts/train/purified_pmi/watch_and_eval.sh"

log() { echo "[$(date -Is)] $*"; }

olmo_status() {
  squeue -u "${USER}" -h -o '%j %T' 2>/dev/null | awk '
    $1=="pmi_olmo7bt_solution" || $1=="pmi_olmo7bt_answer" {print $1,$2}
  '
}

both_gone_or_running() {
  local lines n_pd n_run n_tot
  lines=$(olmo_status || true)
  if [[ -z "${lines}" ]]; then
    # Not in queue — either done or never submitted; allow resubmit.
    return 0
  fi
  n_tot=$(echo "${lines}" | wc -l)
  n_pd=$(echo "${lines}" | grep -c ' PENDING' || true)
  n_run=$(echo "${lines}" | grep -c ' RUNNING' || true)
  # Proceed once no PENDING left (all RUNNING or finished).
  [[ "${n_pd}" -eq 0 ]]
}

log "waiting for olmo PMI trains to leave PENDING..."
while ! both_gone_or_running; do
  log "olmo still pending: $(olmo_status | tr '\n' '; ')"
  sleep "${INTERVAL}"
done
log "olmo cleared PENDING: $(olmo_status | tr '\n' '; ' || echo none)"

find_final() {
  local model_tag="$1" run_name="$2"
  local root="${BASE_DIR}/outputs/${model_tag}/${run_name}"
  local d
  for d in $(ls -1d "${root}"/*/ 2>/dev/null | sort -V -r); do
    if [[ -f "${d}final/adapter_config.json" ]]; then
      echo "${d}final"; return 0
    fi
  done
  return 1
}

resubmit_one() {
  local model_key="$1" priv="$2" model_tag="$3" run_name="$4"
  local ckpt
  ckpt=$(find_final "${model_tag}" "${run_name}") || {
    log "SKIP missing final for ${model_key}/${priv}"; return 0
  }
  log "RESUBMIT ${model_key}/${priv} -> ${ckpt}"
  MODEL_KEY="${model_key}" CHECKPOINT_PATH="${ckpt}" \
    EVAL_TAG="pmi_${priv}_lora_${model_tag}_final" \
    bash "${SUBMIT}"
  date -Is >"${BASE_DIR}/log/train/purified_pmi/watch_state/${model_key}_${priv}.done"
}

resubmit_one qwen3_4b answer qwen3_4b pmi_answer_lora_lr5e6_ot_qwen3_4b
resubmit_one qwen3_4b solution qwen3_4b pmi_solution_lora_lr5e6_ot_qwen3_4b
resubmit_one qwen3_4b_thinking answer qwen3_4b_thinking pmi_answer_lora_lr5e6_ot_qwen3_4b_thinking
resubmit_one qwen3_4b_thinking solution qwen3_4b_thinking pmi_solution_lora_lr5e6_ot_qwen3_4b_thinking

log "restarting pmi_watch_eval for remaining olmo runs"
sbatch "${WATCH}"

log "done"
