#!/bin/bash
#SBATCH --job-name=sft_ocr_tok_4b
#SBATCH --output=log/train_code/sft/4b_base/preprocess_%x.%j.out
#SBATCH --partition=C64M256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=122880M
#SBATCH --time=24:00:00
set -euo pipefail

# Qwen3-1.7B-base and 4B-base share the tokenizer / think chat template.
# Do not tokenize a second copy — reuse ocr.cot.think.le12k.tok.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
echo "[tokenize] 4B reuses the 1.7B tok dir; forwarding to qwen3_1.7b_base/tokenize_ocr_cot_le12k.sh"
exec bash "${BASE_DIR}/scripts/sft_code/qwen3_1.7b_base/tokenize_ocr_cot_le12k.sh"
