#!/bin/bash
#SBATCH --job-name=st_tt_cot_s1024_1p7b
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

# Multi-seed train wrapper (seed=1024). Parent: cot_t4096 (same21981 + CoT privilege).
# Eval these ckpts with default SEED=42.
export SEED=1024
export RUN_NAME=st_tt_clip005_1e_6_openthoughts_cot_t4096_seed1024_1p7b
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"
bash "${BASE_DIR}/scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_cot_t4096.sh"
