#!/bin/bash
#SBATCH --job-name=eval_falcon7b_a25
#SBATCH --output=log/eval/falcon_h1r_7b/aime25/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# SGLang spawns many procs; cluster soft nproc=200 causes EAGAIN under node packing.
ulimit -u 8192 2>/dev/null || true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
echo "[eval] nproc_limit=$(ulimit -u) OMP=${OMP_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS}"

# Falcon-H1R-7B via SGLang in conda env `falcon` (Hybrid Transformer+Mamba2).
# Official sampling (https://huggingface.co/tiiuae/Falcon-H1R-7B):
#   temperature=0.6, top_p=0.95, max_new_tokens=65536,
#   SGLang --reasoning-parser deepseek-r1.
#   Single GPU (tp=1). Disable piecewise CUDA graph: SGLang 0.5.10 warmup
#   hits Dynamo KeyError on Falcon-H1 Mamba layers.
#   max-model-len reduced from the 262144 default to fit A800 KV cache.
# Project deviation: val_n=8 (not the paper's larger n / TTS).
# Submit from OPSD_Analysis so #SBATCH --output lands under log/eval/falcon_h1r_7b.
DATASET=aime25
MODEL_TAG=falcon_h1r_7b
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/falcon-h1r-7b

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

# Official max new tokens is 65536. context_length must be strictly larger than
# prompt + max_new (eval script clamps otherwise). 8k headroom covers chat template.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-65536}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-73728}
VAL_N=${VAL_N:-8}
PASS_AT_K=${PASS_AT_K:-1,4,8}
TP_SIZE=${TP_SIZE:-1}
GENERATE_BATCH_SIZE=${GENERATE_BATCH_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}

cd "${BASE_DIR}"
set +u
source activate falcon
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/vendor/verl:${BASE_DIR}/eval:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1
export SGLANG_DISABLE_TQDM=1
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval/falcon_h1r_7b/${DATASET}"

if [[ ! -f "${CHECKPOINT_PATH}/config.json" ]]; then
  echo "error: checkpoint missing: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}/model-00001-of-00004.safetensors" && ! -f "${CHECKPOINT_PATH}/model.safetensors" ]]; then
  echo "error: incomplete weights in ${CHECKPOINT_PATH}" >&2
  ls -lah "${CHECKPOINT_PATH}" >&2 || true
  exit 1
fi

python -c "import math_verify" >/dev/null
python -c "import mamba_ssm, causal_conv1d" >/dev/null

# Chat template has no enable_thinking switch; default system prompt already
# asks the model to wrap CoT in <think></think>.
THINK_ARGS=(--enable-thinking)

echo "[eval] backend=sglang attention=triton model=Falcon-H1R-7B"
echo "[eval] base_dir=${BASE_DIR}"
echo "[eval] checkpoint=${CHECKPOINT_PATH}"
echo "[eval] dataset=${DATASET} conda=falcon reasoning_parser=deepseek-r1 tp=${TP_SIZE}"
echo "[eval] sampling=official temp=0.6 top_p=0.95 max_new_tokens=${MAX_NEW_TOKENS}"
echo "[eval] context_length=${CONTEXT_LENGTH} val_n=${VAL_N} gen_bs=${GENERATE_BATCH_SIZE}"
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
  --sampling-backend pytorch \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --context-length "${CONTEXT_LENGTH}" \
  --tp-size "${TP_SIZE}" \
  --reasoning-parser deepseek-r1 \
  --disable-piecewise-cuda-graph \
  --resume \
  "${THINK_ARGS[@]}"
