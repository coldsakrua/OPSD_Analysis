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

bash "${script}"
status=$?

# Rename Slurm out: tr_s{seed}_{1p7b|olmo7bt}_st_tt_{variant}.{A}_{a}.out
case "${model_key}" in
  qwen3_1.7b) model_short=1p7b ;;
  olmo3_7b_think) model_short=olmo7bt ;;
  *) model_short="${model_key//./p}" ;;
esac
# OPSD mode tag from run_name (st_tt / snt_tnt / ...)
mode=opsd
if [[ "${run_name}" == st_tt_* ]]; then mode=st_tt
elif [[ "${run_name}" == snt_tnt_* ]]; then mode=snt_tnt
fi
variant=unk
for v in first256 uni256 last256 answer ios c256 c1024; do
  if [[ "${run_name}" == *"_${v}_"* || "${run_name}" == *"_${v}" ]]; then
    variant="${v}"
    break
  fi
done
seed_tok="s?"
if [[ "${run_name}" =~ seed([0-9]+) ]]; then
  seed_tok="s${BASH_REMATCH[1]}"
elif [[ "${SLURM_JOB_NAME}" =~ tr_s([0-9]+) ]]; then
  seed_tok="s${BASH_REMATCH[1]}"
fi
slurm_log="log/train/robustness/array/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${idx}.out"
desc_log="log/train/robustness/array/tr_${seed_tok}_${model_short}_${mode}_${variant}.${SLURM_ARRAY_JOB_ID}_${idx}.out"
if [[ -f "${slurm_log}" && "${slurm_log}" != "${desc_log}" ]]; then
  mv -f "${slurm_log}" "${desc_log}" 2>/dev/null || true
  echo "[array-train] renamed log -> ${desc_log}"
fi

exit "${status}"
