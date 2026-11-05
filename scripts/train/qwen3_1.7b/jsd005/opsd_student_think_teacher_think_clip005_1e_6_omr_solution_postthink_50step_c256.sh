#!/bin/bash
#SBATCH --job-name=st_tt_omr_sol_c256_1p7b
#SBATCH --output=log/train/1.7b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=1:00:00
set -euo pipefail

# Qwen3-1.7B OMR solution_postthink 50step with rollout max_completion=256 (default 1024).
export MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-256}
export RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_omr_solution_postthink_50step_1p7b_c256}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_1p7b_fullparam_50step_omr_solution_postthink_c256}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_omr_solution_postthink_50step.sh
