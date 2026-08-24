#!/bin/bash
#SBATCH --job-name=st_tt_clip005_1e6_ot_cotlen_hard_4b_c256
#SBATCH --output=log/train/4b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=3:00:00
#SBATCH --exclude=gpua800n11,gpua800n13
set -euo pipefail

# 4B st_tt OT cotlen-hard (D7-9) with max_completion=256.
# Keep samples/update = 32 (micro=4, gas=4, 2gpu); max_steps=100 → 3200 samples.
export JSD_TOKEN_CLIP=0.05
export MAX_COMPLETION_LENGTH=256
export PER_DEVICE_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=4
export TARGET_GLOBAL_BATCH=32
export MAX_STEPS=100
export SAVE_STEPS=25
export RUN_NAME=st_tt_clip005_1e_6_ot_cotlen_hard_4b_c256
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_4b_fullparam_100step_ot_cotlen_hard_sttt_c256}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/qwen3_4b/jsd005/st_tt_acc/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_cotlen_hard.sh
