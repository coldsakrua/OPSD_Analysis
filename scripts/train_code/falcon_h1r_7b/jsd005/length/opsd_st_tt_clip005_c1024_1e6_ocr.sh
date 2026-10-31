#!/bin/bash
#SBATCH --job-name=st_tt_clip005_c1024_ocr_falcon7b
#SBATCH --output=log/train_code/falcon_h1r_7b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=12:00:00
set -euo pipefail

# Falcon-H1R-7B OCR: max_completion=1024; keep gbs=32 (micro=2,gas=4,4gpu) — match math.
export JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}
export MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-1024}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
export TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH:-32}
export MAX_STEPS=${MAX_STEPS:-100}
export SAVE_STEPS=${SAVE_STEPS:-25}
export RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_opencodereasoning_falcon7b_c1024}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-falcon_h1r_7b_fullparam_100step_opencodereasoning_c1024}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train_code/falcon_h1r_7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh
