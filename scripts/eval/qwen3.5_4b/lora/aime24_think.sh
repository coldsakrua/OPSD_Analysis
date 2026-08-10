#!/bin/bash
#SBATCH --job-name=st_tt_lora_clip005_lr5e6_q35_4b_a24_th
#SBATCH --output=log/eval/qwen3.5_4b/lora/aime24/think/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --exclude=gpua800n03,gpua800n10,gpua800n05,gpua800n07,gpua800n08
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
set -euo pipefail

# LoRA-run eval for Qwen3.5-4B via SGLang.
# NOTE: eval_math_sglang_local.py cannot load PEFT adapters directly.
# Set CHECKPOINT_PATH to a *merged* full model directory (base+LoRA), not adapter-only.
# Logs under log/eval/qwen3.5_4b/lora/ to distinguish from full-param evals.

THINKING=1
DATASET=aime24
MODEL_TAG=qwen35_4b
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/qwen35_4b

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
fi

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to merged full model dir (SGLang cannot load adapter-only LoRA)}"
EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-${BASE_DIR}/data}
: "${EVAL_DATA_ROOT:?Set EVAL_DATA_ROOT to the server test-data root}"

if [[ -z "${EVAL_TAG:-}" ]]; then
  _ckpt_base="$(basename "${CHECKPOINT_PATH}")"
  if [[ "${_ckpt_base}" == checkpoint-* ]]; then
    _step="${_ckpt_base#checkpoint-}"
    _run="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
    EVAL_TAG="${_run}_ckpt${_step}"
  elif [[ "${_ckpt_base}" == "final" || "${_ckpt_base}" == "merged" ]]; then
    _run="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
    EVAL_TAG="${_run}_${_ckpt_base}"
  else
    EVAL_TAG="$(basename "${CHECKPOINT_PATH}")"
  fi
fi
if [[ "${EVAL_TAG}" != *lora* ]]; then
  EVAL_TAG="lora_${EVAL_TAG}"
fi

OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_think.json}

cd "${BASE_DIR}"
set +u
source activate qwen3_5
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
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
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval/qwen3.5_4b/lora/aime24/think"

python -c "import math_verify" >/dev/null

THINK_ARGS=(--no-thinking)
if [[ "${THINKING}" == "1" ]]; then
  THINK_ARGS=(--enable-thinking)
fi

echo "[eval] backend=sglang attention=triton lora_tag=1 (merged checkpoint required)"
echo "[eval] base_dir=${BASE_DIR}"
echo "[eval] checkpoint=${CHECKPOINT_PATH}"
echo "[eval] dataset=${DATASET} thinking=${THINKING}"
echo "[eval] eval_tag=${EVAL_TAG}"
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
  --generate-batch-size 1 \
  --resume \
  "${THINK_ARGS[@]}"
