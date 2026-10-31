#!/bin/bash
#SBATCH --job-name=he_deepseek_r1_1.5b_th
#SBATCH --output=log/eval_code/deepseek_r1_1.5b/humanevalplus/think/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#
# EvalPlus humanevalplus, deepseek_r1_1.5b — GPU generate only (val-n=4); CPU judge auto-submitted.
# Same sampling as scripts/eval_code/deepseek_r1_1.5b/lcb_v6_*.sh
set -euo pipefail

SEED=${SEED:-42}
THINKING=1
DATASET=humanevalplus
MODEL_TAG=deepseek_r1_1.5b
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/deepseek-r1-distill-qwen-1.5b
GENERATE_BATCH_SIZE=${GENERATE_BATCH_SIZE:-12}

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
DATA_JSONL=${DATA_JSONL:-${BASE_DIR}/data/humanevalplus/HumanEvalPlus.jsonl}

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
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval_code/deepseek_r1_1.5b/humanevalplus/think"

echo "[evalplus] base_dir=${BASE_DIR}"
echo "[evalplus] checkpoint=${CHECKPOINT_PATH}"
echo "[evalplus] dataset=${DATASET} thinking=think (generate-only)"
echo "[evalplus] generate_batch_size=${GENERATE_BATCH_SIZE} (val-n=4)"
echo "[evalplus] output=${OUTPUT_JSON}"
echo "[evalplus] conda_prefix=${CONDA_PREFIX}"
command -v nvidia-smi >/dev/null && nvidia-smi || true

python "${BASE_DIR}/eval_code/eval_evalplus_vllm_local.py" \
  --model-path "${CHECKPOINT_PATH}" \
  --dataset "${DATASET}" \
  --data-jsonl "${DATA_JSONL}" \
  --output-json "${OUTPUT_JSON}" \
  --val-n 4 \
  --pass-at-k 1,4 \
  --max-new-tokens 32768 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k -1 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --seed "${SEED}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 33792 \
  --generate-batch-size "${GENERATE_BATCH_SIZE}" \
  --skip-evaluate \
  --enable-thinking

# Release GPU ASAP: CPU judge on C64 partition (set AUTO_JUDGE=0 to skip).
PASS_AT_K=${PASS_AT_K:-1,4}
JUDGE_SCRIPT="${BASE_DIR}/scripts/eval_code/deepseek_r1_1.5b/humanevalplus_think_judge.sh"
# shellcheck source=/dev/null
source "${BASE_DIR}/scripts/eval_code/evalplus_auto_submit_judge.sh"
evalplus_auto_submit_judge
