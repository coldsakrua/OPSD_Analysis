#!/bin/bash
#SBATCH --job-name=grpo4b_s100_a24
#SBATCH --output=log/eval/qwen3_4b_base/aime24/think/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

FSDP_ACTOR_DIR=${FSDP_ACTOR_DIR:?}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${FSDP_ACTOR_DIR}/hf_merged}
EVAL_TAG=${EVAL_TAG:-$(basename "$(dirname "${FSDP_ACTOR_DIR}")")_$(basename "$(dirname "$(dirname "${FSDP_ACTOR_DIR}")")")}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "$(dirname "${CHECKPOINT_PATH}")" "log/eval/qwen3_4b_base/aime24/think"

if [[ -f "${CHECKPOINT_PATH}/config.json" ]] && compgen -G "${CHECKPOINT_PATH}/*.safetensors" >/dev/null; then
  echo "[merge] reuse existing: ${CHECKPOINT_PATH}"
else
  echo "[merge] merging FSDP -> HF: ${FSDP_ACTOR_DIR} -> ${CHECKPOINT_PATH}"
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${FSDP_ACTOR_DIR}" \
    --target_dir "${CHECKPOINT_PATH}" \
    --use_cpu_initialization
  echo "[merge] done"
fi

ls -lah "${CHECKPOINT_PATH}" | head -20

export CHECKPOINT_PATH EVAL_TAG BASE_DIR
bash "${SCRIPT_DIR}/aime24_think.sh"
