#!/bin/bash
# Mega array worker for multi-seed evals.
# TASK_MANIFEST columns (tab):
#   model_key  script_dir  script_suffix  checkpoint  eval_tag  seed  dataset  output_json
#
#SBATCH --job-name=seed_eval_mega
#SBATCH --output=log/eval/robustness/array/%x_%A_%a.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

: "${TASK_MANIFEST:?Set TASK_MANIFEST}"
: "${BASE_DIR:?Set BASE_DIR}"
cd "${BASE_DIR}"
mkdir -p log/eval/robustness/array

idx=${SLURM_ARRAY_TASK_ID:?}
mapfile -t ROWS < <(grep -vE '^\s*(#|$)' "${TASK_MANIFEST}" || true)
if (( idx < 0 || idx >= ${#ROWS[@]} )); then
  echo "[error] task ${idx} out of range (n=${#ROWS[@]})" >&2
  exit 1
fi

IFS=$'\t' read -r model_key script_dir script_suffix checkpoint eval_tag seed dataset output_json \
  <<<"${ROWS[$idx]}"

script="${script_dir}/${dataset}${script_suffix}.sh"
export BASE_DIR
export CHECKPOINT_PATH="${checkpoint}"
export EVAL_TAG="${eval_tag}"
export SEED="${seed}"
export OUTPUT_JSON="${output_json}"

echo "[array-eval] A=${SLURM_ARRAY_JOB_ID} a=${idx} model=${model_key} ds=${dataset} seed=${seed}"
echo "[array-eval] ckpt=${checkpoint}"
echo "[array-eval] script=${script}"
echo "[array-eval] OUTPUT_JSON=${OUTPUT_JSON}"

if [[ ! -f "${script}" ]]; then
  echo "[error] missing eval script: ${script}" >&2
  exit 1
fi

exec bash "${script}"
