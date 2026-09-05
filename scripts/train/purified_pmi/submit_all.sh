#!/bin/bash
# Submit Purified OPSD-PMI LoRA jobs:
#   4 models × {answer, solution} = 8 runs
# Paper: https://arxiv.org/abs/2607.02234
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
mkdir -p log/train/purified_pmi

QWEN_SH="${ROOT}/scripts/train/purified_pmi/qwen_lora_pmi.sh"
OLMO_SH="${ROOT}/scripts/train/purified_pmi/olmo_lora_pmi.sh"
chmod +x "${QWEN_SH}" "${OLMO_SH}"

MODEL_ROOT=/gpfs/share/home/2501210611/labShare/2501210611/model
OT_PRE="${ROOT}/data/openthoughts/preprocessed"

submit_qwen() {
  local tag="$1"
  local model="$2"
  local dataset="$3"
  local priv="$4"
  local job="pmi_${tag}_${priv}"
  echo "[submit] ${job}"
  MODEL_PATH="${model}" \
  DATASET_PATH="${dataset}" \
  MODEL_TAG="${tag}" \
  TEACHER_PRIVILEGE_FIELD="${priv}" \
  RUN_NAME="pmi_${priv}_lora_lr5e6_ot_${tag}" \
  sbatch --job-name="${job}" "${QWEN_SH}"
}

submit_olmo() {
  local priv="$1"
  local job="pmi_olmo7bt_${priv}"
  echo "[submit] ${job}"
  TEACHER_PRIVILEGE_FIELD="${priv}" \
  RUN_NAME="pmi_${priv}_lora_lr5e6_ot_olmo3_7b_think" \
  sbatch --job-name="${job}" "${OLMO_SH}"
}

# qwen3-1.7b think
DS_17="${OT_PRE}/openthoughts.opsd.solution.sthink_tthink.maxprompt1024.parquet"
submit_qwen "qwen3_1.7b" "${MODEL_ROOT}/qwen3-1.7b" "${DS_17}" solution
submit_qwen "qwen3_1.7b" "${MODEL_ROOT}/qwen3-1.7b" "${DS_17}" answer

# qwen3-4b think
DS_4B="${OT_PRE}/openthoughts.opsd.solution.sthink_tthink.maxprompt1024.parquet"
submit_qwen "qwen3_4b" "${MODEL_ROOT}/qwen3-4b" "${DS_4B}" solution
submit_qwen "qwen3_4b" "${MODEL_ROOT}/qwen3-4b" "${DS_4B}" answer

# qwen3-4b-thinking (2507)
DS_4BT="${OT_PRE}/openthoughts.opsd.solution.sthink_tthink.qwen3_4b_thinking.maxprompt1024.parquet"
submit_qwen "qwen3_4b_thinking" "${MODEL_ROOT}/qwen3-4b-thinking" "${DS_4BT}" solution
submit_qwen "qwen3_4b_thinking" "${MODEL_ROOT}/qwen3-4b-thinking" "${DS_4BT}" answer

# olmo3-7b-think
submit_olmo solution
submit_olmo answer

echo "[done] submitted 8 Purified OPSD-PMI LoRA jobs"
squeue -u "${USER}" | head -40
