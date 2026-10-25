#!/bin/bash
# Sourced by per-model LCB generate scripts.
# Requires: BASE_DIR, OUTPUT_JSON, DATA_JSONL, JUDGE_SCRIPT
# Optional: PASS_AT_K (default 1,4), AUTO_JUDGE (default 1), JUDGE_EXTRA_SBATCH
lcb_auto_submit_judge() {
  if [[ "${AUTO_JUDGE:-1}" != "1" ]]; then
    echo "[lcb] AUTO_JUDGE=0 → skip judge submit"
    return 0
  fi
  : "${JUDGE_SCRIPT:?Set JUDGE_SCRIPT to this model *_judge.sh}"
  local gens="${OUTPUT_JSON%.json}.generations.json"
  # Do NOT pass LCB_PASS_AT_K via --export: Slurm splits on commas, so
  # LCB_PASS_AT_K=1,4 becomes LCB_PASS_AT_K=1. Judge scripts default to 1,4.
  if [[ ! -f "${gens}" ]]; then
    echo "[lcb] ERROR: generations missing, cannot submit judge: ${gens}" >&2
    return 1
  fi
  if [[ ! -f "${JUDGE_SCRIPT}" ]]; then
    echo "[lcb] ERROR: missing JUDGE_SCRIPT=${JUDGE_SCRIPT}" >&2
    return 1
  fi
  echo "[lcb] submitting CPU judge: ${JUDGE_SCRIPT}"
  echo "[lcb] generations=${gens}"
  # Drop any inherited PASS_AT_K so judge script default (1,4) applies.
  # Never put "1,4" in --export (Slurm splits on commas → only "1").
  unset LCB_PASS_AT_K PASS_AT_K 2>/dev/null || true
  # shellcheck source=/dev/null
  source "${BASE_DIR}/scripts/eval_code/judge_job_name.sh"
  local judge_name
  judge_name="$(code_eval_trained_judge_job_name lcb || true)"
  if [[ -n "${judge_name}" ]]; then
    echo "[lcb] judge job-name=${judge_name}"
  fi
  sbatch \
    --export=ALL,BASE_DIR="${BASE_DIR}",LCB_GENERATIONS_JSON="${gens}",LCB_OUTPUT_JSON="${OUTPUT_JSON}",LCB_DATA_JSONL="${DATA_JSONL}" \
    --chdir="${BASE_DIR}" \
    ${JUDGE_EXTRA_SBATCH:-} \
    ${judge_name:+--job-name="${judge_name}"} \
    "${JUDGE_SCRIPT}"
}
