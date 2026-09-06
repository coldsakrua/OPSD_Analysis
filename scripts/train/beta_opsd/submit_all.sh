#!/bin/bash
# Submit paper-main β-OPSD LoRA jobs for 5 models (global_batch=32).
# Paper: https://arxiv.org/abs/2607.28582
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
mkdir -p log/train/beta_opsd

QWEN_SH="${ROOT}/scripts/train/beta_opsd/qwen_lora_beta_opsd.sh"
OLMO_SH="${ROOT}/scripts/train/beta_opsd/olmo_lora_beta_opsd.sh"
chmod +x "${QWEN_SH}" "${OLMO_SH}"

MODEL_ROOT=/gpfs/share/home/2501210611/labShare/2501210611/model
OT_PRE="${ROOT}/data/openthoughts/preprocessed"

submit_qwen() {
  local tag="$1"
  local model="$2"
  local dataset="$3"
  local s_think="$4"
  local t_think="$5"
  local job="beta_${tag}"
  echo "[submit] ${job}"
  MODEL_PATH="${model}" \
  DATASET_PATH="${dataset}" \
  MODEL_TAG="${tag}" \
  STUDENT_THINKING="${s_think}" \
  TEACHER_THINKING="${t_think}" \
  RUN_NAME="beta_opsd_w05to08_rtg099_lora_lr5e6_ot_${tag}" \
  sbatch --job-name="${job}" "${QWEN_SH}"
}

submit_olmo() {
  local tag="$1"
  local model="$2"
  local dataset="$3"
  local s_think="$4"
  local t_think="$5"
  local job="beta_${tag}"
  echo "[submit] ${job}"
  MODEL_PATH="${model}" \
  DATASET_PATH="${dataset}" \
  MODEL_TAG="${tag}" \
  STUDENT_THINKING="${s_think}" \
  TEACHER_THINKING="${t_think}" \
  RUN_NAME="beta_opsd_w05to08_rtg099_lora_lr5e6_ot_${tag}" \
  sbatch --job-name="${job}" "${OLMO_SH}"
}

# qwen3-1.7b think
submit_qwen \
  "qwen3_1.7b" \
  "${MODEL_ROOT}/qwen3-1.7b" \
  "${OT_PRE}/openthoughts.opsd.solution.sthink_tthink.maxprompt1024.parquet" \
  1 1

# qwen3-4b-thinking
submit_qwen \
  "qwen3_4b_thinking" \
  "${MODEL_ROOT}/qwen3-4b-thinking" \
  "${OT_PRE}/openthoughts.opsd.solution.sthink_tthink.qwen3_4b_thinking.maxprompt1024.parquet" \
  1 1

# qwen3-4b-instruct (no native think)
submit_qwen \
  "qwen3_4b_instruct" \
  "${MODEL_ROOT}/qwen3-4b-instruct" \
  "${OT_PRE}/openthoughts.opsd.solution.nothink.instruct.maxprompt1024.parquet" \
  0 0

# olmo3-7b-think
submit_olmo \
  "olmo3_7b_think" \
  "${MODEL_ROOT}/olmo-3-7b-think" \
  "${OT_PRE}/openthoughts.opsd.solution.sthink_tthink.olmo7bthink.maxprompt1024.parquet" \
  1 1

# olmo3-7b-instruct
submit_olmo \
  "olmo3_7b_instruct" \
  "${MODEL_ROOT}/olmo3-7b-it" \
  "${OT_PRE}/openthoughts.opsd.solution.nothink.olmo7bit.maxprompt1024.parquet" \
  0 0

WATCH_SH="${ROOT}/scripts/train/beta_opsd/sbatch_watch_eval.sh"
chmod +x "${WATCH_SH}" "${ROOT}/scripts/train/beta_opsd/watch_and_eval.sh" \
  "${ROOT}/scripts/train/beta_opsd/submit_four_evals.sh" \
  "${ROOT}/scripts/train/beta_opsd/merge_olmo_lora.sh"
echo "[submit] beta_watch_eval (ckpt-200 → 4 evals each)"
sbatch --job-name=beta_watch_eval "${WATCH_SH}"

echo "[done] submitted 5 β-OPSD LoRA trains + 1 watch/eval job"
squeue -u "${USER}" | head -40
