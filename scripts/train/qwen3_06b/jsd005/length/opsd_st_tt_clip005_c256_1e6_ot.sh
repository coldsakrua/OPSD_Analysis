#!/bin/bash
#SBATCH --job-name=st_tt_clip005_c256_06b
#SBATCH --output=log/train/qwen3_06b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpua800n13
set -euo pipefail

# Qwen3-0.6B st_tt + jsd_token_clip=0.05: max_completion=256.
# Match eval stack (conda=anchor / vLLM XFORMERS / enable_thinking).
# Keep samples/update = 128 (same as 0.6B baseline micro=16,gas=4,2gpu):
#   micro=16, gas=4 → gbs=128; max_steps=100 → 12800 samples total.
export JSD_TOKEN_CLIP=0.05
export MAX_COMPLETION_LENGTH=256
export PER_DEVICE_BATCH_SIZE=16
export GRADIENT_ACCUMULATION_STEPS=4
export TARGET_GLOBAL_BATCH=128
export MAX_STEPS=100
export SAVE_STEPS=25
export RUN_NAME=st_tt_clip005_1e_6_ot_06b_c256
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_06b_st_tt_clip005_longgen_c256}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/qwen3_06b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts.sh
