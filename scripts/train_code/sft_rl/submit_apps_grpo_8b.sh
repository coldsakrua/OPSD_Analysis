#!/bin/bash
# Submit Qwen3-8B APPS GRPO (4-GPU). Merge+eval at gs50 once global_step_50 exists
# (see merge_and_eval_gs50_*_apps_*.sh patterns).
#
# Usage (from OPSD_Analysis):
#   bash scripts/train_code/sft_rl/submit_apps_grpo_8b.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

SCRIPT=scripts/train_code/sft_rl/grpo_think_8b_apps_4gpu.sh
MODEL=/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-8b
DATASET=${DATASET_PATH:-${ROOT}/data/apps/preprocessed/apps.traintest.qwen3.think.maxprompt2048.maxtests10.parquet}

mkdir -p log/train_code/sft_rl
chmod +x "${SCRIPT}"

if [[ ! -d "${MODEL}" ]]; then
  echo "[error] missing model: ${MODEL}" >&2
  exit 1
fi
if [[ ! -f "${DATASET}" ]]; then
  echo "[error] missing GRPO APPS dataset: ${DATASET}" >&2
  echo "[error] run: sbatch scripts/train_code/sft_rl/preprocess_apps_grpo.sh" >&2
  exit 1
fi

jid=$(sbatch --parsable --chdir="${ROOT}" "${SCRIPT}")
echo "[submit] grpo_think_8b_apps_4gpu -> jid=${jid}"
echo "[submit] output dir will be: outputs/qwen3_8b/grpo_think_8b_apps_4gpu/${jid}"
echo "[submit] after global_step_50: merge FSDP actor then submit_three_code_evals (MODEL_KEY=qwen3_8b)"
