#!/bin/bash
#SBATCH --job-name=st_tt_clip005_c6144_1p7b
#SBATCH --output=log/train/1.7b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n13
set -euo pipefail

# 1.7B st_tt long-gen + jsd_token_clip=0.05: max_completion=6144.
# Keep samples/update = 64 (same as baseline micro=8,gas=4,2gpu):
#   micro=1, gas=32 → gbs=64; max_steps=100 → 6400 samples total.
export JSD_TOKEN_CLIP=0.05
export MAX_COMPLETION_LENGTH=6144
export PER_DEVICE_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=32
export TARGET_GLOBAL_BATCH=64
export MAX_STEPS=100
export SAVE_STEPS=25
export RUN_NAME=st_tt_clip005_1e_6_ot_1p7b_c6144
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_1p7b_st_tt_clip005_longgen_c6144}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts.sh
