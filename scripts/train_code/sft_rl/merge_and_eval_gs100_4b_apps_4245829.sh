#!/bin/bash
#SBATCH --job-name=merge_eval_gs100_4b_apps
#SBATCH --output=log/train_code/sft_rl/merge_eval_gs100_4b_apps_4245829.%j.out
#SBATCH --partition=C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3:00:00
set -euo pipefail

# Merge FSDP global_step_100 of grpo_think_4b_apps_4gpu (jid 4245829)
# then submit 3 code think evals (humanevalplus / mbppplus / lcb_v6).

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
cd "${BASE_DIR}"

ACTOR="${BASE_DIR}/outputs/qwen3_4b/grpo_think_4b_apps_4gpu/4245829/global_step_100/actor"
MERGED="${ACTOR}/hf_merged"
EVAL_TAG="grpo_think_4b_apps_4gpu_4245829_gs100"
MODEL_KEY="qwen3_4b"

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
  mkdir -p "${MERGED}" "log/train_code/sft_rl"
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${ACTOR}" \
    --target_dir "${MERGED}" \
    --use_cpu_initialization
  hf_ok "${MERGED}" || { echo "merge failed"; ls -lah "${MERGED}" || true; exit 1; }
  echo "[$(date -Is)] merge done"
fi

STATE_FILE="${BASE_DIR}/log/train_code/ocr/watch_state/${MODEL_KEY}_${EVAL_TAG}.submitted"
rm -f "${STATE_FILE}"

MODEL_KEY="${MODEL_KEY}" CHECKPOINT_PATH="${MERGED}" EVAL_TAG="${EVAL_TAG}" \
  SEED=42 BASE_DIR="${BASE_DIR}" AUTO_JUDGE=1 \
  bash "${BASE_DIR}/scripts/train_code/ocr/submit_three_code_evals.sh"

echo "[$(date -Is)] submitted 3 code evals tag=${EVAL_TAG} ckpt=${MERGED}"
