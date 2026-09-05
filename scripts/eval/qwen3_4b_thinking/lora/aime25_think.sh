#!/bin/bash
#SBATCH --job-name=pmi_4bt_a25_lora
#SBATCH --output=log/eval/qwen3_4b_thinking/lora/aime25/think/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail
THINKING=1
DATASET=aime25
MODEL_TAG=4b_thinking
DEFAULT_BASE=/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b-thinking
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then :; elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then BASE_DIR="${SLURM_SUBMIT_DIR}"; else BASE_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"; fi
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to LoRA adapter dir}"
BASE_MODEL_PATH=${BASE_MODEL_PATH:-${DEFAULT_BASE}}
EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-${BASE_DIR}/data}
if [[ -z "${EVAL_TAG:-}" ]]; then
  _ckpt_base="$(basename "${CHECKPOINT_PATH}")"
  _run="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
  EVAL_TAG="${_run}_${_ckpt_base}"
fi
[[ "${EVAL_TAG}" != *lora* && "${EVAL_TAG}" != *pmi* ]] && EVAL_TAG="pmi_lora_${EVAL_TAG}"
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_think.json}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-81920}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-90112}
GENERATE_BATCH_SIZE=${GENERATE_BATCH_SIZE:-4}
cd "${BASE_DIR}"
set +u; source activate anchor; set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/vendor/verl:${BASE_DIR}/eval:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval/qwen3_4b_thinking/lora/${DATASET}/think"
echo "[eval] lora thinking-2507 base=${BASE_MODEL_PATH} adapter=${CHECKPOINT_PATH} ds=${DATASET} tag=${EVAL_TAG}"
python "${BASE_DIR}/eval/eval_math_vllm_local.py" \
  --model-path "${BASE_MODEL_PATH}" --lora-path "${CHECKPOINT_PATH}" \
  --data-root "${EVAL_DATA_ROOT}" --data-format auto --output-json "${OUTPUT_JSON}" \
  --dataset "${DATASET}" --num-samples 0 --val-n 8 --pass-at-k 1,4,8 \
  --max-new-tokens "${MAX_NEW_TOKENS}" --temperature 0.6 --top-p 0.95 --top-k 20 \
  --min-p 0.0 --presence-penalty 0.0 --seed 42 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 --max-model-len "${MAX_MODEL_LEN}" \
  --generate-batch-size "${GENERATE_BATCH_SIZE}" --disable-custom-all-reduce \
  --force-base-tokenizer --enable-thinking
