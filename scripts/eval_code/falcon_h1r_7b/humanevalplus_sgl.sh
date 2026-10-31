#!/bin/bash
#SBATCH --job-name=he_falcon7b_th
#SBATCH --output=log/eval_code/falcon_h1r_7b/humanevalplus/sgl/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#
# EvalPlus humanevalplus, falcon_h1r_7b via SGLang — GPU generate only; CPU judge auto-submitted.
set -euo pipefail

ulimit -u 8192 2>/dev/null || true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

SEED=${SEED:-42}
THINKING=1
DATASET=humanevalplus
MODEL_TAG=falcon_h1r_7b
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/falcon-h1r-7b
GENERATE_BATCH_SIZE=${GENERATE_BATCH_SIZE:-4}

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
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_sgl.json}
DATA_JSONL=${DATA_JSONL:-${BASE_DIR}/data/humanevalplus/HumanEvalPlus.jsonl}

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
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval_code/falcon_h1r_7b/humanevalplus/sgl"

echo "[evalplus] backend=sglang (generate-only)"
echo "[evalplus] base_dir=${BASE_DIR}"
echo "[evalplus] checkpoint=${CHECKPOINT_PATH}"
echo "[evalplus] dataset=${DATASET} thinking=on (template CoT; enable_thinking no-op)"
echo "[evalplus] generate_batch_size=${GENERATE_BATCH_SIZE} (val-n=4)"
echo "[evalplus] output=${OUTPUT_JSON}"
echo "[evalplus] conda_prefix=${CONDA_PREFIX}"
command -v nvidia-smi >/dev/null && nvidia-smi || true
python -c "import mamba_ssm, causal_conv1d" >/dev/null

python "${BASE_DIR}/eval_code/eval_evalplus_sglang_local.py" \
  --model-path "${CHECKPOINT_PATH}" \
  --dataset "${DATASET}" \
  --data-jsonl "${DATA_JSONL}" \
  --output-json "${OUTPUT_JSON}" \
  --val-n 4 \
  --pass-at-k 1,4 \
  --max-new-tokens 65536 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k -1 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --seed "${SEED}" \
  --generate-batch-size "${GENERATE_BATCH_SIZE}" \
  --attention-backend triton \
  --sampling-backend pytorch \
  --mem-fraction-static 0.85 \
  --context-length 73728 \
  --reasoning-parser deepseek-r1 \
  --disable-piecewise-cuda-graph \
  --skip-evaluate \
  --enable-thinking

PASS_AT_K=${PASS_AT_K:-1,4}
JUDGE_SCRIPT="${BASE_DIR}/scripts/eval_code/falcon_h1r_7b/humanevalplus_sgl_judge.sh"
# shellcheck source=/dev/null
source "${BASE_DIR}/scripts/eval_code/evalplus_auto_submit_judge.sh"
evalplus_auto_submit_judge
