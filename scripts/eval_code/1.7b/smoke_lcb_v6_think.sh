#!/bin/bash
#SBATCH --job-name=lcb_17b_smoke
#SBATCH --output=log/eval_code/1.7b/smoke/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#
# Smoke: 1 LCB v6 problem, 1 sample, Qwen3-1.7B think hyperparams (math-aligned).
set -euo pipefail

SEED=${SEED:-42}
DATASET=lcb_v6
MODEL_TAG=1.7b
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${DEFAULT_CKPT}}
EVAL_TAG=${EVAL_TAG:-$(basename "${CHECKPOINT_PATH}")}
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_think.smoke.json}
DATA_JSONL=${DATA_JSONL:-${BASE_DIR}/data/livecodebench/test6.jsonl}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/eval_code:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
ulimit -s 65536
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval_code/1.7b/smoke"

echo "[lcb-smoke] checkpoint=${CHECKPOINT_PATH}"
echo "[lcb-smoke] 1 problem, val-n=1, think params"
command -v nvidia-smi >/dev/null && nvidia-smi || true

python "${BASE_DIR}/eval_code/eval_lcb_vllm_local.py" \
  --model-path "${CHECKPOINT_PATH}" \
  --data-jsonl "${DATA_JSONL}" \
  --output-json "${OUTPUT_JSON}" \
  --max-problems 1 \
  --val-n 1 \
  --pass-at-k 1 \
  --max-new-tokens 38912 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --seed "${SEED}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 40960 \
  --generate-batch-size 1 \
  --enable-thinking
