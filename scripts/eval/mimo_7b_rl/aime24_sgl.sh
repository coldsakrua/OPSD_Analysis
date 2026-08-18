#!/bin/bash
#SBATCH --job-name=eval_mimo7b_aime24_sgl
#SBATCH --output=log/eval/mimo_7b_rl/aime24/sgl/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# Xiaomi MiMo-7B-RL via SGLang (native mimo / mimo_mtp support in sglang>=0.5.x).
# Official eval settings (HF / paper / ModelScope XiaomiMiMo/MiMo-7B-RL):
#   temperature=0.6, top_p=0.95, max_tokens=32768 (math), empty system prompt,
#   AIME24/25 averaged over 32 samples; trust_remote_code.
# https://huggingface.co/XiaomiMiMo/MiMo-7B-RL
# https://www.modelscope.cn/models/XiaomiMiMo/MiMo-7B-RL
DATASET=aime24
MODEL_TAG=mimo_7b_rl
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/mimo-7b-rl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_sgl.json}

# Model config max_position_embeddings=32768. Prompt+max_new must fit; eval script clamps.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32768}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-32768}
# Project eval: n=8 for pass@1,4,8 (official paper uses 32; override with VAL_N=32 if needed).
VAL_N=${VAL_N:-8}
PASS_AT_K=${PASS_AT_K:-1,4,8}
GENERATE_BATCH_SIZE=${GENERATE_BATCH_SIZE:-4}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}

cd "${BASE_DIR}"
set +u
source activate sglang
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/vendor/verl:${BASE_DIR}/eval:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval/mimo_7b_rl/aime24/sgl"

# math-verify must be preinstalled in conda env `sglang` (compute nodes have no PyPI).
python -c "import math_verify" >/dev/null

# Qwen2-style chat template has no enable_thinking switch; RL model reasons in freeform.
THINK_ARGS=(--no-thinking)

echo "[eval] backend=sglang attention=triton model=MiMo-7B-RL"
echo "[eval] base_dir=${BASE_DIR}"
echo "[eval] checkpoint=${CHECKPOINT_PATH}"
echo "[eval] dataset=${DATASET} (empty system prompt; official temp=0.6 top_p=0.95 max_tokens=32768 n=8)"
echo "[eval] max_new_tokens=${MAX_NEW_TOKENS} context_length=${CONTEXT_LENGTH} val_n=${VAL_N} gen_bs=${GENERATE_BATCH_SIZE}"
echo "[eval] output=${OUTPUT_JSON}"
echo "[eval] conda_prefix=${CONDA_PREFIX}"

python "${BASE_DIR}/eval/eval_math_sglang_local.py" \
  --model-path "${CHECKPOINT_PATH}" \
  --data-root "${EVAL_DATA_ROOT}" \
  --data-format auto \
  --output-json "${OUTPUT_JSON}" \
  --dataset "${DATASET}" \
  --num-samples 0 \
  --val-n "${VAL_N}" \
  --pass-at-k "${PASS_AT_K}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k -1 \
  --generate-batch-size "${GENERATE_BATCH_SIZE}" \
  --seed 42 \
  --attention-backend triton \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --context-length "${CONTEXT_LENGTH}" \
  --resume \
  "${THINK_ARGS[@]}"
