#!/bin/bash
#SBATCH --job-name=lcb_v6_nt
#SBATCH --output=log/eval_code/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#
# LiveCodeBench v6 — hyperparams aligned with math nothink eval:
#   val-n=4, pass@1/4, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=32768
#
# Usage (from OPSD_Analysis root):
#   CHECKPOINT_PATH=/path/to/hf_ckpt sbatch scripts/eval_code/run_lcb_v6_nothink.sh
#
set -euo pipefail

DATASET=lcb_v6
MODEL_TAG=${MODEL_TAG:-lcb}
GENERATE_BATCH_SIZE=${GENERATE_BATCH_SIZE:-4}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a full saved model directory}"
EVAL_TAG=${EVAL_TAG:-$(basename "${CHECKPOINT_PATH}")}
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_nothink.json}
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
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval_code"

echo "[lcb] base_dir=${BASE_DIR}"
echo "[lcb] checkpoint=${CHECKPOINT_PATH}"
echo "[lcb] data=${DATA_JSONL}"
echo "[lcb] output=${OUTPUT_JSON}"
echo "[lcb] generate_batch_size=${GENERATE_BATCH_SIZE}"
echo "[lcb] conda=${CONDA_PREFIX}"
command -v nvidia-smi >/dev/null && nvidia-smi || true

python "${BASE_DIR}/eval_code/eval_lcb_vllm_local.py" \
  --model-path "${CHECKPOINT_PATH}" \
  --data-jsonl "${DATA_JSONL}" \
  --output-json "${OUTPUT_JSON}" \
  --val-n 4 \
  --pass-at-k 1,4 \
  --max-new-tokens 32768 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --seed 42 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --generate-batch-size "${GENERATE_BATCH_SIZE}" \
  --skip-evaluate \
  --no-thinking

