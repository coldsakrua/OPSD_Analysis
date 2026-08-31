#!/bin/bash
#SBATCH --job-name=snt_tnt_rkl_clip005_c256_olmo7bi
#SBATCH --output=log/train/olmo3-7b-instruct/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpua800n03,gpua800n10,gpua800n13,gpua800n21
set -euo pipefail

# olmo3_7b_instruct snt_tnt + reverse KL (beta=1.0) + clip=0.05: max_completion=256.
# Keep samples/update = 64 (baseline micro=4,gas=4,4gpu):
#   micro=4, gas=4 → gbs=64; max_steps=100.
export JSD_TOKEN_CLIP=0.05
export BETA=1.0
export MAX_COMPLETION_LENGTH=256
export PER_DEVICE_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=4
export TARGET_GLOBAL_BATCH=64
export MAX_STEPS=100
export SAVE_STEPS=25
export RUN_NAME=snt_tnt_rkl_clip005_1e_6_ot_olmo7bi_c256
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-olmo3_7bi_snt_tnt_rkl_clip005_longgen_c256}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/olmo3_7b_instruct/rkl005/opsd_student_nothink_teacher_nothink_rkl_clip005_1e_6_openthoughts.sh
