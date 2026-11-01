#!/bin/bash
#SBATCH --job-name=smoke_falcon_lcb_p
#SBATCH --output=log/eval_code/smoke/falcon_h1r_7b_lcb_prompt.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#
# Smoke Falcon LCB prompt fix: few problems, check </think> + ```python extract.
set -euo pipefail

ulimit -u 8192 2>/dev/null || true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

SEED=${SEED:-42}
MAX_PROBLEMS=${MAX_PROBLEMS:-3}
VAL_N=${VAL_N:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16384}
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
EVAL_TAG=${EVAL_TAG:-smoke_falcon_lcb_prompt}
OUT_DIR="${BASE_DIR}/eval_outputs/${EVAL_TAG}"
OUT_JSON="${OUT_DIR}/lcb_v6_${MODEL_TAG}_sgl.smoke.json"
GENS_JSON="${OUT_JSON%.json}.generations.json"
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

echo "[smoke] offline prompt checks"
python "${BASE_DIR}/eval_code/smoke_lcb_prompt.py"

echo "[smoke] GPU LCB: problems=${MAX_PROBLEMS} val_n=${VAL_N} max_new=${MAX_NEW_TOKENS}"
python -c "import mamba_ssm, causal_conv1d" >/dev/null
command -v nvidia-smi >/dev/null && nvidia-smi || true

# Fresh smoke gens (do not resume old wrong-prompt outputs).
rm -f "${GENS_JSON}" "${OUT_JSON%.json}.generations.jsonl" "${OUT_JSON}"

python "${BASE_DIR}/eval_code/eval_lcb_sglang_local.py" \
  --model-path "${CHECKPOINT_PATH}" \
  --data-jsonl "${BASE_DIR}/data/livecodebench/test6.jsonl" \
  --output-json "${OUT_JSON}" \
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
  --tp-size 1 \
  --reasoning-parser deepseek-r1 \
  --disable-piecewise-cuda-graph \
  --skip-evaluate \
  --enable-thinking

python "${BASE_DIR}/eval_code/smoke_extract_report.py" \
  --generations-json "${GENS_JSON}" \
  --dataset lcb_v6 \
  --model-path "${CHECKPOINT_PATH}" \
  --max-show "${MAX_PROBLEMS}"

echo "[smoke] falcon LCB prompt smoke done → ${OUT_DIR}"
