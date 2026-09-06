#!/bin/bash
# Slurm job-array worker: task 0..3 → aime24/aime25/aime26/hmmt25.
# Submit via sbatch --array=0-3 (or rely on the directive below).
#
# Required exports:
#   ARRAY_SCRIPT_DIR     dir with per-dataset scripts
#   ARRAY_SCRIPT_SUFFIX  e.g. _think / _sgl / _nothink  → ${ds}${suffix}.sh
# Optional:
#   ARRAY_DATASETS       comma-separated (default aime24,aime25,aime26,hmmt25)
#   ARRAY_OUTPUT_JSON_FMT  path template with __DS__ placeholder for dataset
#   BASE_DIR, CHECKPOINT_PATH, EVAL_TAG, SEED, ... (passed through to child)
#
#SBATCH --job-name=eval_four_array
#SBATCH --output=log/eval/array/%x_%A_%a.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --array=0-3
set -euo pipefail

: "${ARRAY_SCRIPT_DIR:?Set ARRAY_SCRIPT_DIR}"
: "${ARRAY_SCRIPT_SUFFIX:?Set ARRAY_SCRIPT_SUFFIX (e.g. _think)}"

IFS=',' read -r -a DATASETS <<< "${ARRAY_DATASETS:-aime24,aime25,aime26,hmmt25}"
idx=${SLURM_ARRAY_TASK_ID:?}
if (( idx < 0 || idx >= ${#DATASETS[@]} )); then
  echo "[error] SLURM_ARRAY_TASK_ID=${idx} out of range (n=${#DATASETS[@]})" >&2
  exit 1
fi
ds="${DATASETS[$idx]}"
script="${ARRAY_SCRIPT_DIR}/${ds}${ARRAY_SCRIPT_SUFFIX}.sh"
if [[ ! -f "${script}" ]]; then
  echo "[error] missing eval script: ${script}" >&2
  exit 1
fi

if [[ -n "${ARRAY_OUTPUT_JSON_FMT:-}" ]]; then
  export OUTPUT_JSON="${ARRAY_OUTPUT_JSON_FMT//__DS__/${ds}}"
fi

echo "[array] job=${SLURM_ARRAY_JOB_ID:-?} task=${idx} dataset=${ds}"
echo "[array] script=${script}"
echo "[array] OUTPUT_JSON=${OUTPUT_JSON:-<(unset)>}"

# Child scripts already contain #SBATCH headers; running via bash ignores them
# and uses this array task's allocated resources.
exec bash "${script}"
