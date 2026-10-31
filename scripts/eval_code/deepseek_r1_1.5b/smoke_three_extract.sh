#!/bin/bash
#SBATCH --job-name=smoke_r115b_code
#SBATCH --output=log/eval_code/smoke/deepseek_r1_1.5b.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#
# Smoke extract format: LCB + HumanEval+ + MBPP+ (few problems, val-n=1).
set -euo pipefail

SEED=${SEED:-42}
MAX_PROBLEMS=${MAX_PROBLEMS:-2}
VAL_N=${VAL_N:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-12288}
MODEL_TAG=deepseek_r1_1.5b
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/deepseek-r1-distill-qwen-1.5b

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${DEFAULT_CKPT}}
EVAL_TAG=${EVAL_TAG:-smoke_${MODEL_TAG}}
OUT_DIR="${BASE_DIR}/eval_outputs/${EVAL_TAG}"

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
mkdir -p "${OUT_DIR}" "log/eval_code/smoke"

echo "[smoke] model=${CHECKPOINT_PATH} problems=${MAX_PROBLEMS} val_n=${VAL_N} max_new=${MAX_NEW_TOKENS}"
command -v nvidia-smi >/dev/null && nvidia-smi || true

run_lcb() {
  local out="${OUT_DIR}/lcb_v6_${MODEL_TAG}_think.smoke.json"
  local gens="${out%.json}.generations.json"
  python "${BASE_DIR}/eval_code/eval_lcb_vllm_local.py" \
    --model-path "${CHECKPOINT_PATH}" \
    --data-jsonl "${BASE_DIR}/data/livecodebench/test6.jsonl" \
    --output-json "${out}" \
    --max-problems "${MAX_PROBLEMS}" \
    --val-n "${VAL_N}" \
    --pass-at-k 1 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k -1 \
    --min-p 0.0 \
    --presence-penalty 0.0 \
    --seed "${SEED}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --max-model-len $((MAX_NEW_TOKENS + 2048)) \
    --generate-batch-size 1 \
    --skip-evaluate \
    --enable-thinking
  python "${BASE_DIR}/eval_code/smoke_extract_report.py" \
    --generations-json "${gens}" \
    --dataset lcb_v6 \
    --model-path "${CHECKPOINT_PATH}"
}

run_evalplus() {
  local ds=$1
  local jsonl=$2
  local out="${OUT_DIR}/${ds}_${MODEL_TAG}_think.smoke.json"
  local gens="${out%.json}.generations.json"
  python "${BASE_DIR}/eval_code/eval_evalplus_vllm_local.py" \
    --model-path "${CHECKPOINT_PATH}" \
    --dataset "${ds}" \
    --data-jsonl "${jsonl}" \
    --output-json "${out}" \
    --max-problems "${MAX_PROBLEMS}" \
    --val-n "${VAL_N}" \
    --pass-at-k 1 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k -1 \
    --min-p 0.0 \
    --presence-penalty 0.0 \
    --seed "${SEED}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --max-model-len $((MAX_NEW_TOKENS + 2048)) \
    --generate-batch-size 1 \
    --skip-evaluate \
    --enable-thinking
  python "${BASE_DIR}/eval_code/smoke_extract_report.py" \
    --generations-json "${gens}" \
    --dataset "${ds}" \
    --model-path "${CHECKPOINT_PATH}"
}

echo "===== LCB ====="
run_lcb
echo "===== HumanEval+ ====="
run_evalplus humanevalplus "${BASE_DIR}/data/humanevalplus/HumanEvalPlus.jsonl"
echo "===== MBPP+ ====="
run_evalplus mbppplus "${BASE_DIR}/data/mbppplus/MbppPlus.jsonl"
echo "[smoke] deepseek_r1_1.5b done"
