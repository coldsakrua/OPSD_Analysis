#!/bin/bash
set -euo pipefail

THINKING=${1:?thinking flag is required}
BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a full saved model directory}"
: "${EVAL_DATA_ROOT:?Set EVAL_DATA_ROOT to the server test-data root}"
EVAL_TAG=${EVAL_TAG:-$(basename "${CHECKPOINT_PATH}")}
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}_think${THINKING}.json}

cd "${BASE_DIR}"
source activate anchor
export PYTHONPATH="${BASE_DIR}/vendor/verl:${BASE_DIR}/eval:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "${OUTPUT_JSON}")"

THINK_ARGS=(--no-thinking)
if [[ "${THINKING}" == "1" ]]; then
  THINK_ARGS=(--enable-thinking)
fi

python "${BASE_DIR}/eval/eval_math_vllm_local.py" \
  --model-path "${CHECKPOINT_PATH}" \
  --data-root "${EVAL_DATA_ROOT}" \
  --data-format auto \
  --output-json "${OUTPUT_JSON}" \
  --dataset aime24 \
  --dataset aime25 \
  --dataset aime26 \
  --dataset hmmt25 \
  --dataset math500 \
  --num-samples 0 \
  --val-n 16 \
  --pass-at-k 1,4,8,16 \
  --max-new-tokens 32768 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --seed 42 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 40960 \
  --generate-batch-size 16 \
  --disable-custom-all-reduce \
  --force-base-tokenizer \
  "${THINK_ARGS[@]}"
