#!/bin/bash
#SBATCH --job-name=merge_gs100_1p7b_4706799
#SBATCH --output=log/train/sft_rl/merge_gs100_1p7b_sft15000_4706799.%j.out
#SBATCH --partition=C64M512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2:00:00
set -euo pipefail

# Merge FSDP global_step_100 of grpo_think_1p7b_sft15000_4gpu_nooverlong (jid 4706799)
# into actor/hf_merged (required before GRPO@100 → OPSD).

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
cd "${BASE_DIR}"

ACTOR="${BASE_DIR}/outputs/qwen3_1.7b_base/grpo_think_1p7b_sft15000_4gpu_nooverlong/4706799/global_step_100/actor"
MERGED="${ACTOR}/hf_merged"

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
  exit 0
fi

echo "[$(date -Is)] merging FSDP -> HF"
set +u
# shellcheck disable=SC1091
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/vendor/verl:${BASE_DIR}/src:${PYTHONPATH:-}"
mkdir -p "${MERGED}" "log/train/sft_rl"
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR}" \
  --target_dir "${MERGED}" \
  --use_cpu_initialization
hf_ok "${MERGED}" || { echo "merge failed"; ls -lah "${MERGED}" || true; exit 1; }
echo "[$(date -Is)] merge done → ${MERGED}"
