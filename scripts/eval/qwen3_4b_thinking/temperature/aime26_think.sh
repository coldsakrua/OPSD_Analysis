#!/bin/bash
#SBATCH --job-name=eval_4bt_t10_aime26_th
#SBATCH --output=log/eval/qwen3_4b_thinking/temperature/aime26/think/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# Qwen3-4B-Thinking-2507 eval with temperature=1.0 (ablation vs default 0.6).
THINKING=1
DATASET=aime26
MODEL_TAG=4b_thinking
TEMPERATURE=${TEMPERATURE:-1.0}
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b-thinking

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
fi
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${DEFAULT_CKPT}}
EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-${BASE_DIR}/data}
: "${EVAL_DATA_ROOT:?Set EVAL_DATA_ROOT to the server test-data root}"
EVAL_TAG=${EVAL_TAG:-$(basename "${CHECKPOINT_PATH}")}
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_think.json}

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-81920}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-90112}
GENERATE_BATCH_SIZE=${GENERATE_BATCH_SIZE:-4}

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
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval/qwen3_4b_thinking/temperature/aime26/think"

THINK_ARGS=(--enable-thinking)

echo "[eval] temperature=${TEMPERATURE}"
echo "[eval] base_dir=${BASE_DIR}"
echo "[eval] checkpoint=${CHECKPOINT_PATH}"
echo "[eval] dataset=${DATASET} thinking=${THINKING} (Thinking-2507: thinking-only)"
echo "[eval] max_new_tokens=${MAX_NEW_TOKENS} max_model_len=${MAX_MODEL_LEN} gen_bs=${GENERATE_BATCH_SIZE}"
echo "[eval] eval_tag=${EVAL_TAG}"
echo "[eval] output=${OUTPUT_JSON}"
echo "[eval] conda_prefix=${CONDA_PREFIX}"

python "${BASE_DIR}/eval/eval_math_vllm_local.py" \
  --model-path "${CHECKPOINT_PATH}" \
  --data-root "${EVAL_DATA_ROOT}" \
  --data-format auto \
  --output-json "${OUTPUT_JSON}" \
  --dataset "${DATASET}" \
  --num-samples 0 \
  --val-n 8 \
  --pass-at-k 1,4,8 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --seed 42 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --generate-batch-size "${GENERATE_BATCH_SIZE}" \
  --disable-custom-all-reduce \
  --force-base-tokenizer \
  "${THINK_ARGS[@]}"
