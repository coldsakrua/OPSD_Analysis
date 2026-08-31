#!/bin/bash
#SBATCH --job-name=st_tt_jsd_clip005_c256_1p7b
#SBATCH --output=log/train/1.7b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpua800n13
set -euo pipefail

# qwen3_1.7b st_tt + symmetric JSD (beta=0.5) + clip=0.05: max_completion=256.
# Keep samples/update = 64 (baseline micro=8,gas=4,2gpu):
#   micro=8, gas=4 → gbs=64; max_steps=100.
export JSD_TOKEN_CLIP=0.05
export BETA=0.5
export MAX_COMPLETION_LENGTH=256
export PER_DEVICE_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=4
export TARGET_GLOBAL_BATCH=64
export MAX_STEPS=100
export SAVE_STEPS=25
export RUN_NAME=st_tt_jsd_clip005_1e_6_ot_1p7b_c256
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_1p7b_st_tt_jsd_clip005_longgen_c256}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/qwen3_1.7b/jsd_sym005/opsd_student_think_teacher_think_jsd_clip005_1e_6_openthoughts.sh
