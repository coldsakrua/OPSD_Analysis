#!/bin/bash
#SBATCH --job-name=smoke_falcon7b_code
#SBATCH --output=log/eval_code/smoke/falcon_h1r_7b.%j.out
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

ulimit -u 8192 2>/dev/null || true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

SEED=${SEED:-42}
MAX_PROBLEMS=${MAX_PROBLEMS:-2}
VAL_N=${VAL_N:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-12288}
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
EVAL_TAG=${EVAL_TAG:-smoke_${MODEL_TAG}}
OUT_DIR="${BASE_DIR}/eval_outputs/${EVAL_TAG}"
CTX=$((MAX_NEW_TOKENS + 4096))

cd "${BASE_DIR}"
set +u
source activate falcon
set -u
# shellcheck source=/dev/null
source "${BASE_DIR}/scripts/eval_code/sglang_load_gcc.sh"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/eval_code:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
ulimit -s 65536
mkdir -p "${OUT_DIR}" "log/eval_code/smoke"

echo "[smoke] model=${CHECKPOINT_PATH} problems=${MAX_PROBLEMS} val_n=${VAL_N} max_new=${MAX_NEW_TOKENS}"
python -c "import mamba_ssm, causal_conv1d" >/dev/null
command -v nvidia-smi >/dev/null && nvidia-smi || true

run_lcb() {
  local out="${OUT_DIR}/lcb_v6_${MODEL_TAG}_sgl.smoke.json"
  local gens="${out%.json}.generations.json"
  python "${BASE_DIR}/eval_code/eval_lcb_sglang_local.py" \
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
    --generate-batch-size 1 \
    --attention-backend triton \
    --sampling-backend pytorch \
    --mem-fraction-static 0.85 \
    --context-length "${CTX}" \
    --reasoning-parser deepseek-r1 \
    --disable-piecewise-cuda-graph \
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
  local out="${OUT_DIR}/${ds}_${MODEL_TAG}_sgl.smoke.json"
  local gens="${out%.json}.generations.json"
  python "${BASE_DIR}/eval_code/eval_evalplus_sglang_local.py" \
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
    --generate-batch-size 1 \
    --attention-backend triton \
    --sampling-backend pytorch \
    --mem-fraction-static 0.85 \
    --context-length "${CTX}" \
    --reasoning-parser deepseek-r1 \
    --disable-piecewise-cuda-graph \
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
echo "[smoke] falcon_h1r_7b done"
