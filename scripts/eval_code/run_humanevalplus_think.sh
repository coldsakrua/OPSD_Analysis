#!/bin/bash
#SBATCH --job-name=he_th
#SBATCH --output=log/eval_code/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#
# EvalPlus humanevalplus GPU generate (val-n=4). Set JUDGE_SCRIPT to auto-submit CPU judge.
# Usage: CHECKPOINT_PATH=... [JUDGE_SCRIPT=scripts/eval_code/<model>/humanevalplus_think_judge.sh] sbatch ...
#
set -euo pipefail

THINKING=1
DATASET=humanevalplus
MODEL_TAG=${MODEL_TAG:-evalplus}
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
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval_code"

echo "[evalplus] generate-only dataset=${DATASET} thinking=think"
echo "[evalplus] checkpoint=${CHECKPOINT_PATH}"
echo "[evalplus] output=${OUTPUT_JSON}"
command -v nvidia-smi >/dev/null && nvidia-smi || true

python "${BASE_DIR}/eval_code/eval_evalplus_vllm_local.py" \
  --model-path "${CHECKPOINT_PATH}" \
  --dataset "${DATASET}" \
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
  --seed 42 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 40960 \
  --generate-batch-size "${GENERATE_BATCH_SIZE}" \
  --skip-evaluate \
  --enable-thinking

PASS_AT_K=${PASS_AT_K:-1,4}
if [[ -n "${JUDGE_SCRIPT:-}" ]]; then
  # shellcheck source=/dev/null
  source "${BASE_DIR}/scripts/eval_code/evalplus_auto_submit_judge.sh"
  evalplus_auto_submit_judge
else
  echo "[evalplus] JUDGE_SCRIPT unset → generate-only; submit a *_judge.sh manually"
fi
