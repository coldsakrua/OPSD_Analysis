#!/bin/bash
#SBATCH --job-name=merge_eval_gs50_6g
#SBATCH --output=log/train/rl/qwen3_8b_think/merge_eval_gs50_6gpu_bs36_32k_1800.%j.out
#SBATCH --partition=C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3:00:00
set -euo pipefail

# Merge FSDP global_step_50 of grpo_think_8b_omr_int_6gpu_bs36_32k_1800 (jid 4107473)
# then submit 4 think evals (aime24/25/26 + hmmt25).

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
cd "${BASE_DIR}"

ACTOR="${BASE_DIR}/outputs/qwen3_8b/grpo_think_8b_omr_int_6gpu_bs36_32k_1800/4107473/global_step_50/actor"
MERGED="${ACTOR}/hf_merged"
EVAL_TAG="grpo_think_8b_omr_int_6gpu_bs36_32k_1800_gs50"

hf_ok() {
  local p=$1
  [[ -f "${p}/config.json" ]] || return 1
  [[ -f "${p}/model.safetensors" ]] \
    || [[ -f "${p}/model.safetensors.index.json" ]] \
    || [[ -f "${p}/model-00001-of-00001.safetensors" ]] \
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

echo "[$(date -Is)] actor=${ACTOR}"
if [[ ! -d "${ACTOR}" ]]; then
  echo "[error] missing actor dir: ${ACTOR}" >&2
  exit 1
fi

if hf_ok "${MERGED}"; then
  echo "[$(date -Is)] reuse ${MERGED}"
else
  echo "[$(date -Is)] merging FSDP -> HF"
  set +u
  # shellcheck disable=SC1091
  source activate anchor
  set -u
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PYTHONPATH="${BASE_DIR}/vendor/verl:${BASE_DIR}/src:${PYTHONPATH:-}"
  mkdir -p "${MERGED}"
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${ACTOR}" \
    --target_dir "${MERGED}" \
    --use_cpu_initialization
  hf_ok "${MERGED}" || { echo "merge failed"; ls -lah "${MERGED}" || true; exit 1; }
  echo "[$(date -Is)] merge done"
fi

CHECKPOINT_PATH="${MERGED}" EVAL_TAG="${EVAL_TAG}" SEED=42 BASE_DIR="${BASE_DIR}" \
  bash "${BASE_DIR}/scripts/eval/8b/submit_four.sh"

echo "[$(date -Is)] submitted 4 think evals tag=${EVAL_TAG}"
