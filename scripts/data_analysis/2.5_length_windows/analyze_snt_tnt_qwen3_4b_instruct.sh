#!/bin/bash
#SBATCH --job-name=da25snttntqwen34binstruct
#SBATCH --output=log/data_analysis/2.5_length_windows/qwen3_4b_instruct/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# 2.5 length-window analysis snt_tnt qwen3_4b_instruct
# Rollout length: 6144 (match length training; override MAX_COMPLETION).
# Length windows: 0-128, 128-256, 256-512, 512-1024, 1024-2048, 2048-4096, 4096-6144.
# Metrics: JSD KL, top-k KL (k=1,16), log-ratio, loss-dominant tokens.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
export TASK=length_windows
export MODEL_KEY=qwen3_4b_instruct
export COMBO=snt_tnt
export MAX_COMPLETION=6144

source "${BASE_DIR}/scripts/data_analysis/_common_launch.sh"
