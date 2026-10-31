#!/bin/bash
#SBATCH --job-name=lcb_06b_v6_th
#SBATCH --output=log/eval_code/0.6b/lcb_v6/think/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#
# LiveCodeBench v6, Qwen3-1.7B — same sampling as scripts/eval/0.6b/*_think.sh
set -euo pipefail

SEED=${SEED:-42}
THINKING=1
DATASET=lcb_v6
MODEL_TAG=0.6b
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-0.6b
GENERATE_BATCH_SIZE=${GENERATE_BATCH_SIZE:-16}

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
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_think.json}
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
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval_code/0.6b/lcb_v6/think"

echo "[lcb] base_dir=${BASE_DIR}"
echo "[lcb] checkpoint=${CHECKPOINT_PATH}"
echo "[lcb] dataset=${DATASET} thinking=${THINKING}"
echo "[lcb] generate_batch_size=${GENERATE_BATCH_SIZE} (val-n=4 → ~$((GENERATE_BATCH_SIZE * 4)) seqs @12k)"
echo "[lcb] output=${OUTPUT_JSON}"
echo "[lcb] conda_prefix=${CONDA_PREFIX}"
command -v nvidia-smi >/dev/null && nvidia-smi || true

python "${BASE_DIR}/eval_code/eval_lcb_vllm_local.py" \
  --model-path "${CHECKPOINT_PATH}" \
  --data-jsonl "${DATA_JSONL}" \
  --output-json "${OUTPUT_JSON}" \
  --val-n 4 \
  --pass-at-k 1,4 \
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
  --generate-batch-size "${GENERATE_BATCH_SIZE}" \
  --skip-evaluate \
  --enable-thinking


# Release GPU ASAP: CPU judge on C64 partition (set AUTO_JUDGE=0 to skip).
PASS_AT_K=${PASS_AT_K:-1,4}
JUDGE_SCRIPT="${BASE_DIR}/scripts/eval_code/0.6b/lcb_v6_think_judge.sh"
# shellcheck source=/dev/null
source "${BASE_DIR}/scripts/eval_code/lcb_auto_submit_judge.sh"
lcb_auto_submit_judge
