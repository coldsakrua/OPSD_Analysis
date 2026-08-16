#!/bin/bash
#SBATCH --job-name=snt_tnt_clip005_1e6_ot_cotlen_hard_c256_olmo7bi
#SBATCH --output=log/train/olmo3-7b-instruct/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=03:00:00
#SBATCH --exclude=gpua800n03,gpua800n10,gpua800n13,gpua800n21
set -euo pipefail

# Olmo-3-7B-Instruct snt_tnt OT cotlen-hard (D7-9) with max_completion=256.
# Keep samples/update = 64 (micro=4, gas=4, 4gpu); max_steps=100 → 6400 samples.
export JSD_TOKEN_CLIP=0.05
export MAX_COMPLETION_LENGTH=256
export PER_DEVICE_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=4
export TARGET_GLOBAL_BATCH=64
export MAX_STEPS=100
export SAVE_STEPS=25
export RUN_NAME=snt_tnt_clip005_1e_6_ot_cotlen_hard_olmo7bit_c256
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-olmo3_7b_instruct_fullparam_100step_ot_cotlen_hard_c256}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/olmo3_7b_instruct/jsd005/snt_tnt_acc/opsd_student_nothink_teacher_nothink_clip005_1e_6_openthoughts_cotlen_hard.sh
