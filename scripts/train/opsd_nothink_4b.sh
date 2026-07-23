#!/bin/bash
#SBATCH --job-name=opsd_nothink_4b
#SBATCH --output=log/opsd_%x.%j.out
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=440G
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n06,gpua800n09,gpua800n11,gpua800n12,gpua800n13,gpua800n16
set -euo pipefail

MODE=correct
THINKING=0
RUN_NAME=opsd_nothink_4b

# Slurm copies the batch script to /var/spool; prefer submit dir over BASH_SOURCE.
BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/dapo/preprocessed/dapo-math-17k.opsd.correct.nothink.maxprompt1024.parquet}
: "${DATASET_PATH:?Set DATASET_PATH to the server-side DAPO parquet path}"
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}

cd "${BASE_DIR}"
# conda activate scripts reference unset vars; keep nounset elsewhere
set +u
source activate anchor
set -u

export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export WANDB_MODE=offline
export WANDB_PROJECT=${WANDB_PROJECT:-OPSD}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_4b_fullparam_100step}
export WANDB_DIR=${WANDB_DIR:-${BASE_DIR}/wandb}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_CONFIGURE_LOGGING=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HYDRA_FULL_ERROR=1
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_HOME}"

THINK_ARGS=()
if [[ "${THINKING}" == "1" ]]; then
  THINK_ARGS+=(--enable-thinking)
fi

# per-device=2 keeps KL loss under memory; gas=8 raises optimizer-step batch to 64.
# Generate once for the full accumulation window, then accumulate loss slices.
echo "[launch] run=${RUN_NAME} mode=${MODE} thinking=${THINKING}"
echo "[launch] model=${MODEL_PATH} dataset=${DATASET_PATH} output=${OUTPUT_DIR}"
echo "[launch] 4 GPUs, microbatch=2, gas=8, global batch=64, vLLM util=0.45, gen-once-per-step"

accelerate launch \
  --config_file "${BASE_DIR}/configs/accelerate_zero3.yaml" \
  --num_processes 4 \
  --main_process_port "${MASTER_PORT:-29500}" \
  "${BASE_DIR}/src/train_opsd.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --privilege-mode "${MODE}" \
  --max-steps 100 \
  --save-steps 25 \
  --max-prompt-length 1024 \
  --max-completion-length 6144 \
  --per-device-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-6 \
  --vllm-gpu-memory-utilization 0.45 \
  --deepspeed "${BASE_DIR}/configs/deepspeed_zero3.json" \
  "${THINK_ARGS[@]}"
