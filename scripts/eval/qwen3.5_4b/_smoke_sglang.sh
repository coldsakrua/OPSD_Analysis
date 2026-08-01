#!/bin/bash
#SBATCH --job-name=smoke_q35_sgl
#SBATCH --output=log/eval/qwen3.5_4b/smoke/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
set -euo pipefail

cd /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis
mkdir -p log/eval/qwen3.5_4b/smoke

set +u
source activate qwen3_5
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Prefer CUDA 12.8 nvcc when present (FlashInfer JIT needs modern nvcc flags).
if [[ -d /usr/local/cuda-12.8 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
elif command -v module >/dev/null 2>&1; then
  module load cuda/12.8 2>/dev/null || true
fi
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1
export SGLANG_DISABLE_TQDM=1
export PYTHONPATH="${PWD}/vendor/verl:${PWD}/eval:${PYTHONPATH:-}"

python scripts/eval/qwen3.5_4b/_smoke_sglang.py
