#!/bin/bash
#SBATCH --job-name=lcb_falcon7b_v6_4g
#SBATCH --output=log/eval_code/falcon_h1r_7b/lcb_v6/sgl/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#
# LiveCodeBench v6, Falcon-H1R-7B via SGLang, 4-GPU TP.
# Same sampling as lcb_v6_sgl.sh; resumes existing .generations.jsonl when --resume.
set -euo pipefail

ulimit -u 8192 2>/dev/null || true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

SEED=${SEED:-42}
THINKING=1
DATASET=lcb_v6
MODEL_TAG=falcon_h1r_7b
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/falcon-h1r-7b
TP_SIZE=${TP_SIZE:-4}
GENERATE_BATCH_SIZE=${GENERATE_BATCH_SIZE:-8}

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
DATA_JSONL=${DATA_JSONL:-${BASE_DIR}/data/livecodebench/test6.jsonl}

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
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval_code/falcon_h1r_7b/lcb_v6/sgl"

echo "[lcb] backend=sglang attention=triton model=Falcon-H1R-7B"
echo "[lcb] base_dir=${BASE_DIR}"
echo "[lcb] checkpoint=${CHECKPOINT_PATH}"
echo "[lcb] dataset=${DATASET} thinking=${THINKING} tp=${TP_SIZE}"
echo "[lcb] generate_batch_size=${GENERATE_BATCH_SIZE}"
echo "[lcb] output=${OUTPUT_JSON}"
echo "[lcb] conda_prefix=${CONDA_PREFIX}"
command -v nvidia-smi >/dev/null && nvidia-smi || true
python -c "import mamba_ssm, causal_conv1d" >/dev/null

python "${BASE_DIR}/eval_code/eval_lcb_sglang_local.py" \
  --model-path "${CHECKPOINT_PATH}" \
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
  --tp-size "${TP_SIZE}" \
  --reasoning-parser deepseek-r1 \
  --disable-piecewise-cuda-graph \
  --skip-evaluate \
  --resume \
  --enable-thinking

# Release GPU ASAP: CPU judge on C64 partition (set AUTO_JUDGE=0 to skip).
PASS_AT_K=${PASS_AT_K:-1,4}
JUDGE_SCRIPT="${BASE_DIR}/scripts/eval_code/falcon_h1r_7b/lcb_v6_sgl_judge.sh"
# shellcheck source=/dev/null
source "${BASE_DIR}/scripts/eval_code/lcb_auto_submit_judge.sh"
lcb_auto_submit_judge
