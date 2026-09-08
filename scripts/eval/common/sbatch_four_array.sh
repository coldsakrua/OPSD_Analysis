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
bash "${script}"
status=$?

# Rename Slurm out to include model + variant + dataset when possible.
# Expected final: seed{SEED}_{1p7b|olmo7bt}_{variant}_{dataset}.{A}_{a}.out
seed_tok="seed${SEED:-42}"
case "${ARRAY_SCRIPT_DIR}" in
  */eval/1.7b*) model_short=1p7b; log_root="log/eval/1.7b/array" ;;
  */eval/olmo_7b_think*) model_short=olmo7bt; log_root="log/eval/olmo_7b_think/array" ;;
  *) model_short=model; log_root="log/eval/array" ;;
esac
tag="${EVAL_TAG:-unk}"
variant=unk
for v in topk16_rkl topk4_rkl topk1_rkl topk16 topk4 topk1 first256 uni256 last256 irr_other_sol answer c256; do
  if [[ "${tag}" == *"${v}"* ]]; then
    if [[ "${v}" == irr_other_sol ]]; then variant=ios; else variant="${v}"; fi
    break
  fi
done
if [[ "${variant}" == unk && "${tag}" == *openthoughts* ]]; then variant=c1024; fi
if [[ "${variant}" == unk && ( "${tag}" == qwen3-1.7b* || "${tag}" == olmo-3-7b-think* ) ]]; then variant=base; fi
# attach train-seed for multi-seed trained variants when present in tag
if [[ "${tag}" =~ seed([0-9]+) && "${variant}" != unk && "${variant}" != base && "${variant}" != topk* ]]; then
  if [[ "${tag}" == *"${variant}_seed${BASH_REMATCH[1]}"* || "${tag}" == *"seed${BASH_REMATCH[1]}_${variant}"* || "${tag}" == *"_seed${BASH_REMATCH[1]}"* ]]; then
    # only when this looks like a train-seed run name (contains seed before checkpoint)
    if [[ "${tag}" == *"_seed${BASH_REMATCH[1]}_"* || "${tag}" == *"_${variant}_seed${BASH_REMATCH[1]}"* ]]; then
      variant="${variant}_s${BASH_REMATCH[1]}"
    fi
  fi
fi

slurm_log="${log_root}/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${idx}.out"
desc_log="${log_root}/${seed_tok}_${model_short}_${variant}_${ds}.${SLURM_ARRAY_JOB_ID}_${idx}.out"
if [[ -f "${slurm_log}" && "${slurm_log}" != "${desc_log}" ]]; then
  mv -f "${slurm_log}" "${desc_log}" 2>/dev/null || true
  echo "[array] renamed log -> ${desc_log}"
fi

exit "${status}"
