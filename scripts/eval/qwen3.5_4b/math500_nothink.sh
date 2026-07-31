#!/bin/bash
#SBATCH --job-name=eval_q35_4b_math500_nt
#SBATCH --output=log/eval/qwen3.5_4b/math500/nothink/%x.%j.out
#SBATCH --partition=GPUA800
#SBATCH --exclude=gpua800n03,gpua800n10,gpua800n05,gpua800n07,gpua800n08
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
set -euo pipefail

# Qwen3.5-4B via SGLang 0.5.10 (vLLM failed: missing flash-attn CUDA ext on glibc 2.28).
# Official sampling: temp=1.0 top_p=0.95 top_k=20 presence_penalty=1.5; max_new_tokens=81920 (README math/coding contests)
# think: chat_template enable_thinking=True (default); nothink: enable_thinking=False
THINKING=0
DATASET=math500
MODEL_TAG=qwen35_4b
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/qwen35_4b

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Slurm copies the batch script under /var/spool/...; prefer submit dir.
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${DEFAULT_CKPT}}
EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-${BASE_DIR}/data}
: "${EVAL_DATA_ROOT:?Set EVAL_DATA_ROOT to the server test-data root}"
EVAL_TAG=${EVAL_TAG:-$(basename "${CHECKPOINT_PATH}")}
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_nothink.json}

cd "${BASE_DIR}"
# conda activate scripts reference unset vars; keep nounset elsewhere
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
export PYTHONPATH="${BASE_DIR}/vendor/verl:${BASE_DIR}/eval:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1
export SGLANG_DISABLE_TQDM=1
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval/qwen3.5_4b/${DATASET}/nothink"

# math-verify must be preinstalled in conda env qwen3_5 (compute nodes have no PyPI).
python -c "import math_verify" >/dev/null

THINK_ARGS=(--no-thinking)
if [[ "${THINKING}" == "1" ]]; then
  THINK_ARGS=(--enable-thinking)
fi

echo "[eval] backend=sglang attention=triton"
echo "[eval] base_dir=${BASE_DIR}"
echo "[eval] checkpoint=${CHECKPOINT_PATH}"
echo "[eval] dataset=${DATASET} thinking=${THINKING}"
echo "[eval] output=${OUTPUT_JSON}"
echo "[eval] conda_prefix=${CONDA_PREFIX}"

python "${BASE_DIR}/eval/eval_math_sglang_local.py" \
  --model-path "${CHECKPOINT_PATH}" \
  --data-root "${EVAL_DATA_ROOT}" \
  --data-format auto \
  --output-json "${OUTPUT_JSON}" \
  --dataset "${DATASET}" \
  --num-samples 0 \
  --val-n 8 \
  --pass-at-k 1,4,8 \
  --max-new-tokens 81920 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --seed 42 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --mem-fraction-static 0.80 \
  --context-length 131072 \
  --reasoning-parser qwen3 \
  --enable-hybrid-swa-memory \
  --generate-batch-size 4 \
  --resume \
  "${THINK_ARGS[@]}"
