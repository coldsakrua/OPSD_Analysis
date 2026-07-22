#!/bin/bash
#SBATCH --job-name=opsd_pi_th
#SBATCH --output=opsd_%x.%j.out
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n24,gpua800n09,gpua800n26,gpua800n11,gpua800n12,gpua800n16
exec bash "${SLURM_SUBMIT_DIR}/scripts/train_common.sh" pi 1 opsd_pi_think
