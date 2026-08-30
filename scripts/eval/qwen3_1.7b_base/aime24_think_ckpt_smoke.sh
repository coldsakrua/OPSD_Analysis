#!/bin/bash
#SBATCH --job-name=sft_ckpt_a24_smoke
#SBATCH --output=log/eval/qwen3_1.7b_base/aime24/think/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
set -euo pipefail

# Parameterized SFT ckpt smoke on AIME24 (thinking + stop_token_ids).
# Override: CHECKPOINT_PATH, EVAL_TAG, NUM_SAMPLES, VAL_N, GEN_BATCH, JOB name via sbatch -J
THINKING=1
DATASET=aime24
MODEL_TAG=qwen3_1.7b_base
DEFAULT_CKPT=/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/outputs/qwen3_1.7b_base/sft_think_4gpu_12k/3299217/checkpoint-1500

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
EVAL_TAG=${EVAL_TAG:-sft_think_4gpu_12k_ckpt1500_stopfix_smoke}
NUM_SAMPLES=${NUM_SAMPLES:-8}
VAL_N=${VAL_N:-4}
GEN_BATCH=${GEN_BATCH:-4}
OUTPUT_JSON=${OUTPUT_JSON:-${BASE_DIR}/eval_outputs/${EVAL_TAG}/${DATASET}_${MODEL_TAG}_think.json}

cd "${BASE_DIR}"
set +u
source activate anchor
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${BASE_DIR}/vendor/verl:${BASE_DIR}/eval:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
# qwen3-*-base config max_pos=32768; allow 40k for 38912 think evals
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
mkdir -p "$(dirname "${OUTPUT_JSON}")" "log/eval/qwen3_1.7b_base/aime24/think"

THINK_ARGS=(--no-thinking)
if [[ "${THINKING}" == "1" ]]; then
  THINK_ARGS=(--enable-thinking)
fi

PASS_AT_K="1"
if (( VAL_N >= 2 )); then PASS_AT_K="${PASS_AT_K},2"; fi
if (( VAL_N >= 4 )); then PASS_AT_K="${PASS_AT_K},4"; fi
if (( VAL_N >= 8 )); then PASS_AT_K="${PASS_AT_K},8"; fi

echo "[eval] base_dir=${BASE_DIR}"
echo "[eval] checkpoint=${CHECKPOINT_PATH}"
echo "[eval] dataset=${DATASET} thinking=${THINKING} smoke=ckpt"
echo "[eval] num_samples=${NUM_SAMPLES} val_n=${VAL_N} gen_batch=${GEN_BATCH}"
echo "[eval] output=${OUTPUT_JSON}"
echo "[eval] conda_prefix=${CONDA_PREFIX}"

python "${BASE_DIR}/eval/eval_math_vllm_local.py" \
  --model-path "${CHECKPOINT_PATH}" \
  --data-root "${EVAL_DATA_ROOT}" \
  --data-format auto \
  --output-json "${OUTPUT_JSON}" \
  --dataset "${DATASET}" \
  --num-samples "${NUM_SAMPLES}" \
  --val-n "${VAL_N}" \
  --pass-at-k "${PASS_AT_K}" \
  --max-new-tokens 38912 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --seed 42 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 40960 \
  --allow-long-max-model-len \
  --generate-batch-size "${GEN_BATCH}" \
  --disable-custom-all-reduce \
  --force-base-tokenizer \
  "${THINK_ARGS[@]}"
