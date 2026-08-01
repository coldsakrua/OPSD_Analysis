#!/bin/bash
#SBATCH --job-name=snt_oti_g16
#SBATCH --output=log/train/4b-instruct/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n02,gpua800n03,gpua800n04,gpua800n06,gpua800n09,gpua800n10,gpua800n13,gpua800n14
set -euo pipefail

# Single-factor Instruct OPSD hyper variant.
# Overrides: PER_DEVICE_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4
export PER_DEVICE_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=4
export RUN_NAME=snt_tnt_gbs16_oti
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/qwen3_4b_instruct/openthoughts/opsd_student_nothink_teacher_nothink_1e_6_openthoughts.sh
