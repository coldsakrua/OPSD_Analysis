#!/bin/bash
#SBATCH --job-name=ocr_sft_prep
#SBATCH --output=log/train_code/sft/1.7b_base/preprocess_%x.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=187488M
#SBATCH --time=24:00:00
set -euo pipefail

# Wrapper around scripts/data/run_prepare_ocr_sft_cot_le12k.sh

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
mkdir -p "${BASE_DIR}/log/train_code/sft/1.7b_base"
exec bash "${BASE_DIR}/scripts/data/run_prepare_ocr_sft_cot_le12k.sh"
