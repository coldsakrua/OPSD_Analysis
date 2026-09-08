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

# Descriptive log name: ev_s{seed}_{model}_{variant}_{dataset}.{A}_{a}.out
case "${model_key}" in
  qwen3_1.7b) model_short=1p7b ;;
  olmo3_7b_think) model_short=olmo7bt ;;
  qwen3_4b_instruct) model_short=4bit ;;
  *) model_short="${model_key//./p}" ;;
esac
variant=unk
for v in first256 uni256 last256 irr_other_sol answer c256 c1024; do
  if [[ "${eval_tag}" == *"${v}"* ]]; then
    if [[ "${v}" == irr_other_sol ]]; then variant=ios; else variant="${v}"; fi
    break
  fi
done
if [[ "${variant}" == unk && "${eval_tag}" == *openthoughts* ]]; then
  variant=c1024
fi
desc_log="log/eval/robustness/array/ev_s${seed}_${model_short}_${variant}_${dataset}.${SLURM_ARRAY_JOB_ID}_${idx}.out"
# Slurm still writes to --output (%x_%A_%a.out); add a same-dir symlink with model/dataset.
slurm_log="log/eval/robustness/array/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${idx}.out"
if [[ -e "${slurm_log}" || -L "${slurm_log}" || true ]]; then
  ln -sfn "$(basename "${slurm_log}")" "${desc_log}" || true
fi

echo "[array-eval] A=${SLURM_ARRAY_JOB_ID} a=${idx} model=${model_key} ds=${dataset} seed=${seed} variant=${variant}"
echo "[array-eval] ckpt=${checkpoint}"
echo "[array-eval] script=${script}"
echo "[array-eval] OUTPUT_JSON=${OUTPUT_JSON}"
echo "[array-eval] desc_log=${desc_log} -> $(basename "${slurm_log}")"

if [[ ! -f "${script}" ]]; then
  echo "[error] missing eval script: ${script}" >&2
  exit 1
fi

bash "${script}"
# Prefer final descriptive filename once the job finishes (safe if Slurm has closed the handle).
if [[ -f "${slurm_log}" && ! -e "${desc_log}" ]]; then
  mv -f "${slurm_log}" "${desc_log}" || true
elif [[ -f "${slurm_log}" && -L "${desc_log}" ]]; then
  rm -f "${desc_log}"
  mv -f "${slurm_log}" "${desc_log}" || true
fi
