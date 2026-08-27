#!/bin/bash
#SBATCH --exclude=gpua800n13,gpua800n21
#SBATCH --job-name=da25_snt_tnt_olmo7bi
#SBATCH --output=log/data_analysis/25/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# 2.5 length windows snt_tnt on olmo3_7b_instruct
# Length windows: 0-128 … 4096-6144.
# Rollout: temp=1.1 top_p=0.95 top_k=20 max_prompt=1024 max_completion=6144

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/scripts/data_analysis/outputs/length_windows/olmo3_7b_instruct/snt_tnt_${JOB_TAG}}

TASK="length_windows"
MODEL_KEY="olmo3_7b_instruct"
COMBO="snt_tnt"
MODEL_PATH="/gpfs/share/home/2501210611/labShare/2501210611/model/olmo3-7b-it"
CONDA_ENV="sglang"
BACKEND="sglang"
NUM_PROMPTS=${NUM_PROMPTS:-2048}
N_ROLLOUTS=${N_ROLLOUTS:-2}
MAX_PROMPT=${MAX_PROMPT:-1024}
MAX_COMPLETION=${MAX_COMPLETION:-6144}
SCORE_BATCH=${SCORE_BATCH:-1}
GEN_BATCH_HINT=${GEN_BATCH_HINT:-8}

mkdir -p "${OUTPUT_DIR}" "${BASE_DIR}/log/data_analysis/25"

cd "${BASE_DIR}"
set +u
source activate "sglang"
set -u
_NVIDIA_LIB_ROOT="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia"
_NVIDIA_LD=""
if [[ -d "${_NVIDIA_LIB_ROOT}" ]]; then
  for _lib in "${_NVIDIA_LIB_ROOT}"/*/lib; do [[ -d "${_lib}" ]] && _NVIDIA_LD="${_NVIDIA_LD:+${_NVIDIA_LD}:}${_lib}"; done
fi
export LD_LIBRARY_PATH="${_NVIDIA_LD:+${_NVIDIA_LD}:}${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -d /usr/local/cuda-12.6 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.6
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
elif [[ -d /usr/local/cuda-12.8 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
fi
if [[ -n "${_NVIDIA_LD}" ]]; then export LD_LIBRARY_PATH="${_NVIDIA_LD}:${LD_LIBRARY_PATH}"; fi
if command -v module >/dev/null 2>&1; then module load gcc/11 2>/dev/null || module load gcc/9 2>/dev/null || true; fi
export SGLANG_MEM_FRACTION_STATIC="0.8"
export SGLANG_ATTENTION_BACKEND=triton
export SGLANG_SAMPLING_BACKEND=pytorch

export PYTHONPATH="${BASE_DIR}/src:${BASE_DIR}/scripts/data_analysis:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_CONFIGURE_LOGGING=0

EXTRA_ARGS=(
  --task "${TASK}"
  --model-key "${MODEL_KEY}"
  --combo "${COMBO}"
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --num-prompts "${NUM_PROMPTS}"
  --n-rollouts "${N_ROLLOUTS}"
  --max-prompt-length "${MAX_PROMPT}"
  --max-completion-length "${MAX_COMPLETION}"
  --temperature 1.1
  --top-p 0.95
  --top-k 20
  --score-batch-size "${SCORE_BATCH}"
  --gen-batch-hint "${GEN_BATCH_HINT}"
  --backend "${BACKEND}"
  --gpu-memory-utilization 0.90
  --seed 42
)
EXTRA_ARGS+=(--attention-backend triton --sampling-backend pytorch --mem-fraction-static 0.8)

echo "[analysis] task=${TASK} model=${MODEL_KEY} combo=${COMBO} backend=${BACKEND}"
echo "[analysis] output=${OUTPUT_DIR}"

echo "[analysis] ===== phase 1: generate ====="
python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-score

echo "[analysis] ===== phase 2: score ====="
python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-generate

echo "[analysis] done -> ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
