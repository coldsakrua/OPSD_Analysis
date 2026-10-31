#!/bin/bash
#SBATCH --job-name=he_mimo7b_nt
#SBATCH --output=log/eval_code/mimo_7b_rl/humanevalplus/sgl/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#
# EvalPlus humanevalplus, mimo_7b_rl via SGLang — GPU generate only; CPU judge auto-submitted.
set -euo pipefail

SEED=${SEED:-42}
THINKING=0
DATASET=humanevalplus
MODEL_TAG=mimo_7b_rl
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/mimo-7b-rl
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
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_sgl_nothink.json}
DATA_JSONL=${DATA_JSONL:-${BASE_DIR}/data/humanevalplus/HumanEvalPlus.jsonl}

cd "${BASE_DIR}"
set +u
source activate sglang
set -u
# shellcheck source=/dev/null
source "${BASE_DIR}/scripts/eval_code/sglang_load_gcc.sh"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/eval_code:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
ulimit -s 65536
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval_code/mimo_7b_rl/humanevalplus/sgl"

echo "[evalplus] backend=sglang (generate-only)"
echo "[evalplus] base_dir=${BASE_DIR}"
echo "[evalplus] checkpoint=${CHECKPOINT_PATH}"
echo "[evalplus] dataset=${DATASET} thinking=nothink"
echo "[evalplus] generate_batch_size=${GENERATE_BATCH_SIZE} (val-n=4)"
echo "[evalplus] output=${OUTPUT_JSON}"
echo "[evalplus] conda_prefix=${CONDA_PREFIX}"
command -v nvidia-smi >/dev/null && nvidia-smi || true

python "${BASE_DIR}/eval_code/eval_evalplus_sglang_local.py" \
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
  --generate-batch-size "${GENERATE_BATCH_SIZE}" \
  --attention-backend triton \
  --sampling-backend pytorch \
  --mem-fraction-static 0.80 \
  --context-length 32768 \
  --skip-evaluate \
  --no-thinking

PASS_AT_K=${PASS_AT_K:-1,4}
JUDGE_SCRIPT="${BASE_DIR}/scripts/eval_code/mimo_7b_rl/humanevalplus_sgl_nothink_judge.sh"
# shellcheck source=/dev/null
source "${BASE_DIR}/scripts/eval_code/evalplus_auto_submit_judge.sh"
evalplus_auto_submit_judge
