#!/bin/bash
#SBATCH --job-name=st_tt_clip005_c1024_ocr_r1_15b
#SBATCH --output=log/train_code/deepseek_r1_1.5b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpua800n13
set -euo pipefail

# DeepSeek-R1-1.5B OCR: max_completion=1024; keep gbs=64 (micro=8,gas=4,2gpu).
export JSD_TOKEN_CLIP=0.05
export MAX_COMPLETION_LENGTH=1024
export PER_DEVICE_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=4
export TARGET_GLOBAL_BATCH=64
export MAX_STEPS=100
export SAVE_STEPS=25
export RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_opencodereasoning_r1_15b_c1024}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-deepseek_r1_1p5b_fullparam_100step_opencodereasoning_c1024}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train_code/deepseek_r1_1.5b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh
