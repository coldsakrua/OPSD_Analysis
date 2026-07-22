#!/bin/bash
#SBATCH --job-name=opsd_eval_nt
#SBATCH --output=opsd_%x.%j.out
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
exec bash "${SLURM_SUBMIT_DIR}/scripts/eval_common.sh" 0
