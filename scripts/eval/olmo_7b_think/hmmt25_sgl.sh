#!/bin/bash
#SBATCH --job-name=eval_olmo37bt_hmmt25_sgl
#SBATCH --output=log/eval/olmo_7b_think/hmmt25/sgl/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# Olmo-3-7B-Think via SGLang (local: triton attn + pytorch sampling).
# Official recommended settings (allenai/Olmo-3-7B-Think):
#   temperature=0.6, top_p=0.95, max_tokens=32768
# Project N=8 / pass@1,4,8.
THINKING=1
DATASET=hmmt25
MODEL_TAG=olmo_7b_think
DEFAULT_CKPT=/gpfs/share/home/2501210611/labShare/2501210611/model/olmo-3-7b-think

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BASE_DIR:-}" ]]; then
  :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${DEFAULT_CKPT}}
EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-${BASE_DIR}/data}
: "${EVAL_DATA_ROOT:?Set EVAL_DATA_ROOT to the server test-data root}"
EVAL_TAG=${EVAL_TAG:-$(basename "${CHECKPOINT_PATH}")}
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_sgl_think.json}

cd "${BASE_DIR}"
set +u
source activate sglang
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/vendor/verl:${BASE_DIR}/eval:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval/olmo_7b_think/hmmt25/sgl"

python -c "import math_verify" >/dev/null

THINK_ARGS=(--enable-thinking)

echo "[eval] backend=sglang attention=triton model=Olmo-3-7B-Think"
echo "[eval] base_dir=${BASE_DIR}"
echo "[eval] checkpoint=${CHECKPOINT_PATH}"
echo "[eval] dataset=${DATASET} thinking=on (chat_template opens <think>)"
echo "[eval] sampling=official temp=0.6 top_p=0.95 max_new_tokens=32768"
echo "[eval] output=${OUTPUT_JSON}"
echo "[eval] conda_prefix=${CONDA_PREFIX}"

python "${BASE_DIR}/eval/eval_math_sglang_local.py" \
  --model-path "${CHECKPOINT_PATH}" \
  --data-root "${EVAL_DATA_ROOT}" \
  --data-format auto \
  --output-json "${OUTPUT_JSON}" \
  --dataset "${DATASET}" \
  --num-samples 0 \
  --val-n 8 \
  --pass-at-k 1,4,8 \
  --max-new-tokens 32768 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k -1 \
  --generate-batch-size 4 \
  --seed 42 \
  --attention-backend triton \
  --mem-fraction-static 0.80 \
  --context-length 40960 \
  --resume \
  "${THINK_ARGS[@]}"
