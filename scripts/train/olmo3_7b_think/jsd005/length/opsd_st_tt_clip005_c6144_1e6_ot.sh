#!/bin/bash
#SBATCH --job-name=st_tt_clip005_c6144_olmo7bt
#SBATCH --output=log/train/olmo3-7b-think/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=72:00:00
set -euo pipefail

# Olmo-3-7B-Think st_tt + jsd_token_clip=0.05: max_completion=6144.
# Think-only. Keep samples/update = 64 (baseline micro=4,gas=4,4gpu):
#   micro=1, gas=16 → gbs=64; max_steps=100 → 6400 samples total.
export JSD_TOKEN_CLIP=0.05
export MAX_COMPLETION_LENGTH=6144
export PER_DEVICE_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=16
export TARGET_GLOBAL_BATCH=64
export MAX_STEPS=100
export SAVE_STEPS=25
export SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.30}
export RUN_NAME=st_tt_clip005_1e_6_ot_olmo7bt_c6144
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-olmo3_7bt_st_tt_clip005_longgen_c6144}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts.sh
