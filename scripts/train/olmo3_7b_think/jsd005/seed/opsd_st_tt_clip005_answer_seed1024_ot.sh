#!/bin/bash
#SBATCH --job-name=st_tt_answer_s1024_olmo
#SBATCH --output=log/train/olmo3-7b-think/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=03:00:00
#SBATCH --exclude=gpua800n03,gpua800n10,gpua800n13,gpua800n21
set -euo pipefail

# Multi-seed train wrapper (seed=1024). Same hyperparams as parent; only SEED/RUN_NAME differ.
# Parent: scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_answer.sh
# Eval these ckpts with default SEED=42.
# variant=answer
export SEED=1024
export RUN_NAME=st_tt_clip005_answer_seed1024_olmo7bt
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"
bash "/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_answer.sh"
