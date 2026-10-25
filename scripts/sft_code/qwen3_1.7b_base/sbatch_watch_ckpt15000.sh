#!/bin/bash
#SBATCH --job-name=watch_1p7b_sft15k
#SBATCH --output=log/train_code/sft/1.7b_base/watch_ckpt15000.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3906M
#SBATCH --time=2-00:00:00
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
TRAIN_JID=${TRAIN_JID:-4071750}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/outputs/qwen3_1.7b_base/sft_think_4gpu_12k_ocr/3925961}
CKPT_STEP=${CKPT_STEP:-15000}
MODEL_KEY=${MODEL_KEY:-qwen3_1.7b_base}
EVAL_TAG=${EVAL_TAG:-sft_think_4gpu_12k_ocr_ckpt${CKPT_STEP}}
INTERVAL=${INTERVAL:-60}
POST_DONE_WAIT=${POST_DONE_WAIT:-90}

cd "${BASE_DIR}"
mkdir -p log/train_code/sft/1.7b_base

echo "[watch-slurm] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[watch-slurm] train_jid=${TRAIN_JID} ckpt=${CKPT_STEP}"
echo "[watch-slurm] output_dir=${OUTPUT_DIR}"

export BASE_DIR TRAIN_JID OUTPUT_DIR CKPT_STEP MODEL_KEY EVAL_TAG INTERVAL POST_DONE_WAIT
chmod +x "${BASE_DIR}/scripts/sft_code/qwen3_1.7b_base/watch_sft_ckpt_then_eval.sh" \
  "${BASE_DIR}/scripts/train_code/ocr/submit_three_code_evals.sh"
exec bash "${BASE_DIR}/scripts/sft_code/qwen3_1.7b_base/watch_sft_ckpt_then_eval.sh"
