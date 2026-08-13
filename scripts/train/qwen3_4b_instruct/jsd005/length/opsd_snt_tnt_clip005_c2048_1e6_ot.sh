#!/bin/bash
#SBATCH --job-name=snt_tnt_clip005_c2048_4bi
#SBATCH --output=log/train/4b-instruct/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpua800n13
set -euo pipefail

# 4B-Instruct snt_tnt long-gen + jsd_token_clip=0.05: max_completion=2048.
# Instruct has no think mode. Keep samples/update = 32 (baseline micro=4,gas=4,2gpu):
#   micro=2, gas=8 → gbs=32; max_steps=100 → 3200 samples total.
export JSD_TOKEN_CLIP=0.05
export MAX_COMPLETION_LENGTH=2048
export PER_DEVICE_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=8
export TARGET_GLOBAL_BATCH=32
export MAX_STEPS=100
export SAVE_STEPS=25
export RUN_NAME=snt_tnt_clip005_1e_6_oti_c2048
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_4bi_snt_tnt_clip005_longgen_c2048}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/qwen3_4b_instruct/jsd005/opsd_student_nothink_teacher_nothink_clip005_1e_6_openthoughts.sh
