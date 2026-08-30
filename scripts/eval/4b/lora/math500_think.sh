#!/bin/bash
#SBATCH --job-name=st_tt_lora_clip005_lr5e6_4b_m500_th
#SBATCH --output=log/eval/4b/lora/math500/think/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
set -euo pipefail

# LoRA eval (vLLM): CHECKPOINT_PATH = PEFT adapter dir (.../checkpoint-N or .../final).
# BASE_MODEL_PATH defaults to the frozen base weights used in training.
# Logs under log/eval/4b/lora/ to distinguish from full-param evals.

THINKING=1
DATASET=math500
MODEL_TAG=4b
DEFAULT_BASE=/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
fi

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to LoRA adapter directory (…/checkpoint-N or …/final)}"
BASE_MODEL_PATH=${BASE_MODEL_PATH:-${DEFAULT_BASE}}
EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-${BASE_DIR}/data}
: "${EVAL_DATA_ROOT:?Set EVAL_DATA_ROOT to the server test-data root}"

if [[ -z "${EVAL_TAG:-}" ]]; then
  _ckpt_base="$(basename "${CHECKPOINT_PATH}")"
  if [[ "${_ckpt_base}" == checkpoint-* ]]; then
    _step="${_ckpt_base#checkpoint-}"
    _run="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
    EVAL_TAG="${_run}_ckpt${_step}"
  elif [[ "${_ckpt_base}" == "final" ]]; then
    _run="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
    EVAL_TAG="${_run}_final"
  else
    EVAL_TAG="$(basename "${CHECKPOINT_PATH}")"
  fi
fi
# Ensure tag is visually LoRA-distinct even if user overrides run name oddly.
if [[ "${EVAL_TAG}" != *lora* ]]; then
  EVAL_TAG="lora_${EVAL_TAG}"
fi

OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_think.json}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/vendor/verl:${BASE_DIR}/eval:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval/4b/lora/math500/think"

THINK_ARGS=(--no-thinking)
if [[ "${THINKING}" == "1" ]]; then
  THINK_ARGS=(--enable-thinking)
fi

echo "[eval] backend=vllm lora=1"
echo "[eval] base_dir=${BASE_DIR}"
echo "[eval] base_model=${BASE_MODEL_PATH}"
echo "[eval] lora_adapter=${CHECKPOINT_PATH}"
echo "[eval] dataset=${DATASET} thinking=${THINKING}"
echo "[eval] eval_tag=${EVAL_TAG}"
echo "[eval] output=${OUTPUT_JSON}"
echo "[eval] conda_prefix=${CONDA_PREFIX}"

python "${BASE_DIR}/eval/eval_math_vllm_local.py" \
  --model-path "${BASE_MODEL_PATH}" \
  --lora-path "${CHECKPOINT_PATH}" \
  --data-root "${EVAL_DATA_ROOT}" \
  --data-format auto \
  --output-json "${OUTPUT_JSON}" \
  --dataset "${DATASET}" \
  --num-samples 0 \
  --val-n 8 \
  --pass-at-k 1,4,8 \
  --max-new-tokens 38912 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --seed 42 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 40960 \
  --generate-batch-size 8 \
  --disable-custom-all-reduce \
  --force-base-tokenizer \
  "${THINK_ARGS[@]}"
