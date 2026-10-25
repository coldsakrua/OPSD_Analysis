#!/bin/bash
# Sourced by per-model EvalPlus generate scripts.
# Requires: BASE_DIR, OUTPUT_JSON, DATA_JSONL, DATASET, JUDGE_SCRIPT
# Optional: PASS_AT_K (default 1,4), AUTO_JUDGE (default 1), JUDGE_EXTRA_SBATCH
evalplus_auto_submit_judge() {
  if [[ "${AUTO_JUDGE:-1}" != "1" ]]; then
    echo "[evalplus] AUTO_JUDGE=0 → skip judge submit"
    return 0
  fi
  : "${JUDGE_SCRIPT:?Set JUDGE_SCRIPT to this model *_judge.sh}"
  : "${DATASET:?Set DATASET to humanevalplus or mbppplus}"
  local gens="${OUTPUT_JSON%.json}.generations.json"
  # Do NOT pass EVALPLUS_PASS_AT_K via --export: Slurm splits on commas, so
  # EVALPLUS_PASS_AT_K=1,4 becomes EVALPLUS_PASS_AT_K=1. Judge scripts default to 1,4.
  if [[ ! -f "${gens}" ]]; then
    echo "[evalplus] ERROR: generations missing, cannot submit judge: ${gens}" >&2
    return 1
  fi
  if [[ ! -f "${JUDGE_SCRIPT}" ]]; then
    echo "[evalplus] ERROR: missing JUDGE_SCRIPT=${JUDGE_SCRIPT}" >&2
    return 1
  fi
  echo "[evalplus] submitting CPU judge: ${JUDGE_SCRIPT}"
  echo "[evalplus] generations=${gens}"
  # Drop any inherited PASS_AT_K so judge script default (1,4) applies.
  # Never put "1,4" in --export (Slurm splits on commas → only "1").
  unset EVALPLUS_PASS_AT_K PASS_AT_K 2>/dev/null || true
  local ds_short
  case "${DATASET}" in
    humanevalplus) ds_short=he ;;
    mbppplus) ds_short=mb ;;
    *) ds_short="${DATASET}" ;;
  esac
  # shellcheck source=/dev/null
  source "${BASE_DIR}/scripts/eval_code/judge_job_name.sh"
  local judge_name
  judge_name="$(code_eval_trained_judge_job_name "${ds_short}" || true)"
  if [[ -n "${judge_name}" ]]; then
    echo "[evalplus] judge job-name=${judge_name}"
  fi
  sbatch \
    --export=ALL,BASE_DIR="${BASE_DIR}",EVALPLUS_GENERATIONS_JSON="${gens}",EVALPLUS_OUTPUT_JSON="${OUTPUT_JSON}",EVALPLUS_DATA_JSONL="${DATA_JSONL}",EVALPLUS_DATASET="${DATASET}" \
    --chdir="${BASE_DIR}" \
    ${JUDGE_EXTRA_SBATCH:-} \
    ${judge_name:+--job-name="${judge_name}"} \
    "${JUDGE_SCRIPT}"
}
