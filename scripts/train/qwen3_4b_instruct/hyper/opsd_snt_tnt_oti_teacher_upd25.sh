#!/bin/bash
#SBATCH --job-name=snt_oti_tupd25
#SBATCH --output=log/train/4b-instruct/opsd_%x.%j.out
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n02,gpua800n03,gpua800n04,gpua800n06,gpua800n09,gpua800n10,gpua800n13,gpua800n14
set -euo pipefail

# Instruct OPSD hyper variant: periodic teacher (hard-copy student→teacher every 25 steps)
# instead of a fixed initial checkpoint teacher. Other hypers match the base 1e-6 run.
export TEACHER_UPDATE_STEPS=25
export RUN_NAME=snt_tnt_teacher_upd25_oti
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_4b_instruct_teacher_upd25}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/qwen3_4b_instruct/openthoughts/opsd_student_nothink_teacher_nothink_1e_6_openthoughts.sh
