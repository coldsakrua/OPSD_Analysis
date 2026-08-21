#!/bin/bash
#SBATCH --job-name=eval_q25m15_a24
#SBATCH --output=log/eval/qwen25_math_1.5b/aime24/nothink/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
set -euo pipefail

# Qwen2.5-Math-1.5B-Instruct
# Official sampling (ModelScope / README, maj@8 / RM@8):
#   temperature=0.7, top_p=0.8
#   CoT system (chat template default):
#     "Please reason step by step, and put your final answer within \boxed{}."
# Context: max_position_embeddings=4096 (not 32k).
# Instruct has no <think> mode -> --no-thinking.
# Project: val_n=8, pass@1,4,8 (same harness as scripts/eval/deepseek_r1_1.5b).
THINKING=0
DATASET=aime24
MODEL_TAG=qwen25_math_1.5b
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/qwen2.5-math-1.5b-it

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
source activate anchor
set -u
# Prefer conda libstdc++ (GLIBCXX_3.4.29) over system /lib64
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/vendor/verl:${BASE_DIR}/eval:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval/qwen25_math_1.5b/aime24/nothink"

if [[ ! -f "${CHECKPOINT_PATH}/config.json" ]]; then
  echo "error: checkpoint missing: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}/model.safetensors" && ! -f "${CHECKPOINT_PATH}/model-00001-of-00001.safetensors" ]]; then
  echo "error: incomplete weights in ${CHECKPOINT_PATH}" >&2
  ls -lah "${CHECKPOINT_PATH}" >&2 || true
  exit 1
fi

THINK_ARGS=(--no-thinking)
if [[ "${THINKING}" == "1" ]]; then
  THINK_ARGS=(--enable-thinking)
fi

echo "[eval] base_dir=${BASE_DIR}"
echo "[eval] checkpoint=${CHECKPOINT_PATH}"
echo "[eval] dataset=${DATASET} thinking=${THINKING} (Instruct CoT, no think tags)"
echo "[eval] sampling=official temp=0.7 top_p=0.8 top_k=-1 max_new=4096 max_model_len=4096"
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
  --max-new-tokens 4096 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k -1 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --seed 42 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --generate-batch-size 8 \
  --disable-custom-all-reduce \
  --force-base-tokenizer \
  "${THINK_ARGS[@]}"
