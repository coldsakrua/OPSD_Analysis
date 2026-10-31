#!/bin/bash
#SBATCH --job-name=st_tt_clip005_c1024_ocr_06b
#SBATCH --output=log/train_code/0.6b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=12:00:00
set -euo pipefail

# 0.6B st_tt OCR + jsd_token_clip=0.05: max_completion=1024 (explicit c1024 tag).
# Same hyperparams as the baseline OCR script; RUN_NAME includes _c1024 for clarity.
export JSD_TOKEN_CLIP=0.05
export MAX_COMPLETION_LENGTH=1024
export PER_DEVICE_BATCH_SIZE=16
export GRADIENT_ACCUMULATION_STEPS=4
export TARGET_GLOBAL_BATCH=128
export MAX_STEPS=100
export SAVE_STEPS=25
export RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_opencodereasoning_06b_c1024}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_06b_fullparam_100step_opencodereasoning_c1024}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train_code/qwen3_06b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh
