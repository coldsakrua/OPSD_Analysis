#!/bin/bash
#SBATCH --job-name=eval_r1_15b_a25
#SBATCH --output=log/eval/deepseek_r1_1.5b/aime25/think/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
set -euo pipefail

# DeepSeek-R1-Distill-Qwen-1.5B (base: Qwen2.5-Math-1.5B)
# Official sampling (ModelScope / README):
#   temperature=0.6, top_p=0.95, max_new_tokens=32768
#   no system prompt; chat template forces assistant to start with <think>\n
#   math: "Please reason step by step, and put your final answer within \boxed{}."
# Project: val_n=8, pass@1,4,8 (same harness as scripts/eval/1.7b).
THINKING=1
DATASET=aime25
MODEL_TAG=deepseek_r1_1.5b
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/deepseek-r1-distill-qwen-1.5b

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
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_think.json}

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
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval/deepseek_r1_1.5b/aime25/think"

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
echo "[eval] dataset=${DATASET} thinking=${THINKING}"
echo "[eval] sampling=official temp=0.6 top_p=0.95 top_k=-1 max_new=32768"
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
  --max-new-tokens 32768 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k -1 \
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
