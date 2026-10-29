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
# variant prefers short ablation keywords; otherwise compress EVAL_TAG
# (aligned with 1.7b job names / prior manual renames like ot_integer_answer_wrong_ckpt50).
seed_tok="seed${SEED:-42}"
case "${ARRAY_SCRIPT_DIR}" in
  */eval/1.7b*) model_short=1p7b; log_root="log/eval/1.7b/array" ;;
  */eval/olmo_7b_think*) model_short=olmo7bt; log_root="log/eval/olmo_7b_think/array" ;;
  *) model_short=model; log_root="log/eval/array" ;;
esac
tag="${EVAL_TAG:-unk}"
variant=unk
# Old ablation short-names only (exact-ish tokens). Do NOT match bare "answer"
# (would crush ot_integer_answer_* into "answer").
for v in topk16_rkl topk4_rkl topk1_rkl topk16 topk4 topk1 first256 uni256 last256 irr_other_sol c256; do
  if [[ "${tag}" == *"${v}"* ]]; then
    if [[ "${v}" == irr_other_sol ]]; then variant=ios; else variant="${v}"; fi
    break
  fi
done
if [[ "${variant}" == unk && "${tag}" == *answer_s* ]]; then variant=answer; fi
if [[ "${variant}" == unk && "${tag}" == *openthoughts* && "${tag}" != *ot_integer* && "${tag}" != *instr* && "${tag}" != *clip0* && "${tag}" != *clip1* ]]; then
  variant=c1024
fi
if [[ "${variant}" == unk && ( "${tag}" == qwen3-1.7b* || "${tag}" == olmo-3-7b-think* ) ]]; then variant=base; fi
# Fallback: compress EVAL_TAG → distinctive middle (matches 1.7b / manual olmo renames).
if [[ "${variant}" == unk && -n "${EVAL_TAG:-}" && "${EVAL_TAG}" != unk ]]; then
  v="${EVAL_TAG}"
  v="${v//_olmo7bt/}"
  v="${v//_1p7b/}"
  v="${v//_50step/}"
  v="${v//_100step/}"
  v="${v//_30step/}"
  for pfx in st_tt_clip005_1e_6_ st_tt_clip005_1e6_ snt_tnt_clip005_1e_6_ st_tt_clip005_ st_tt_; do
    if [[ "${v}" == "${pfx}"* ]]; then
      v="${v#"${pfx}"}"
      break
    fi
  done
  # Drop trailing eval seed suffix if present (seed is already in filename).
  v="${v%_seed${SEED:-42}}"
  variant="${v:-unk}"
fi
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
elif [[ -f "${desc_log}" ]]; then
  : # already descriptive
else
  # Recover: job may have already been renamed to *_unk_*; rewrite from EVAL_TAG.
  unk_log="${log_root}/${seed_tok}_${model_short}_unk_${ds}.${SLURM_ARRAY_JOB_ID}_${idx}.out"
  if [[ -f "${unk_log}" && "${unk_log}" != "${desc_log}" && "${variant}" != unk ]]; then
    mv -f "${unk_log}" "${desc_log}" 2>/dev/null || true
    echo "[array] renamed log -> ${desc_log}"
  fi
fi

exit "${status}"
