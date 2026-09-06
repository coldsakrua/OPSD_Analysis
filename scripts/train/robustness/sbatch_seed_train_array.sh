#!/bin/bash
# Array worker: TASK_MANIFEST line i → bash that seed train wrapper.
# Manifest columns (tab): model_key model_tag run_name script
#
#SBATCH --job-name=seed_train_arr
#SBATCH --output=log/train/robustness/array/%x_%A_%a.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=12:00:00
set -euo pipefail

: "${TASK_MANIFEST:?Set TASK_MANIFEST}"
: "${BASE_DIR:?Set BASE_DIR}"
cd "${BASE_DIR}"
mkdir -p log/train/robustness/array

idx=${SLURM_ARRAY_TASK_ID:?}
mapfile -t ROWS < <(grep -vE '^\s*(#|$)' "${TASK_MANIFEST}" || true)
if (( idx < 0 || idx >= ${#ROWS[@]} )); then
  echo "[error] task ${idx} out of range (n=${#ROWS[@]})" >&2
  exit 1
fi

IFS=$'\t' read -r model_key model_tag run_name script <<<"${ROWS[$idx]}"
job_tag="${SLURM_ARRAY_JOB_ID}_${idx}"
out_dir="${BASE_DIR}/outputs/${model_tag}/${run_name}/${job_tag}"

echo "[array-train] A=${SLURM_ARRAY_JOB_ID} a=${idx} model=${model_key} run=${run_name}"
echo "[array-train] script=${script}"
echo "[array-train] OUTPUT_DIR=${out_dir}"

if [[ ! -f "${script}" ]]; then
  echo "[error] missing script: ${script}" >&2
  exit 1
fi

# Parent scripts key off OUTPUT_DIR / RUN_NAME; force unique per-task output dir.
export BASE_DIR
export OUTPUT_DIR="${out_dir}"
export RUN_NAME="${run_name}"
# Avoid accidental collision if a parent still uses SLURM_JOB_ID as JOB_TAG.
export JOB_TAG="${job_tag}"

exec bash "${script}"
