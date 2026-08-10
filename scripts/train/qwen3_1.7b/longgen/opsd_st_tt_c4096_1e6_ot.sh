#!/bin/bash
#SBATCH --job-name=st_tt_c4096_1p7b
#SBATCH --output=log/train/1.7b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=48:00:00
#SBATCH --exclude=gpua800n13
set -euo pipefail

# 1.7B st_tt long-gen: max_completion=4096.
# Keep samples/update = 64 (same as baseline micro=8,gas=4,2gpu):
#   micro=2, gas=16 → gbs=64; max_steps=100 → 6400 samples total.
export MAX_COMPLETION_LENGTH=4096
export PER_DEVICE_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=16
export TARGET_GLOBAL_BATCH=64
export MAX_STEPS=100
export SAVE_STEPS=25
export RUN_NAME=st_tt_1e_6_ot_1p7b_c4096
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_1p7b_st_tt_longgen_c4096}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/qwen3_1.7b/openthoughts/opsd_student_think_teacher_think_1e_6_openthoughts.sh
