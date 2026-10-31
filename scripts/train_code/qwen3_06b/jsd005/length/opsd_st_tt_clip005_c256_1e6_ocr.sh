#!/bin/bash
#SBATCH --job-name=st_tt_clip005_c256_ocr_06b
#SBATCH --output=log/train_code/0.6b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=3:00:00
set -euo pipefail

# 0.6B st_tt OCR + jsd_token_clip=0.05: max_completion=256 (baseline default=1024).
# Keep samples/update = 128 (same as baseline micro=16,gas=4,2gpu):
#   micro=16, gas=4 → gbs=128; max_steps=100 → 12800 samples total.
export JSD_TOKEN_CLIP=0.05
export MAX_COMPLETION_LENGTH=256
export PER_DEVICE_BATCH_SIZE=16
export GRADIENT_ACCUMULATION_STEPS=4
export TARGET_GLOBAL_BATCH=128
export MAX_STEPS=100
export SAVE_STEPS=25
export RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_opencodereasoning_06b_c256}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_06b_fullparam_100step_opencodereasoning_c256}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train_code/qwen3_06b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh
