#!/bin/bash
#SBATCH --exclude=gpua800n13,gpua800n21
#SBATCH --job-name=da21_snt_tnt_q35
#SBATCH --output=log/data_analysis/21/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# 2.1 combo snt_tnt on qwen3.5_4b
# Rollout: temp=1.1 top_p=0.95 top_k=20 max_prompt=1024 max_completion=1024

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
# 2.2: POOL=long (default, +sol_long@12288) | POOL=short (legacy sol/answer/irrelevant)
POOL=${POOL:-long}
RUN_SUFFIX="snt_tnt"
if [[ "2.1" == "2.2" && "${POOL}" == "short" ]]; then
  RUN_SUFFIX="snt_tnt_short"
fi
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/scripts/data_analysis/outputs/combinations/qwen3.5_4b/${RUN_SUFFIX}_${JOB_TAG}}

TASK="combinations"
MODEL_KEY="qwen3.5_4b"
COMBO="snt_tnt"
MODEL_PATH="/gpfs/share/home/2501210611/labShare/2501210611/model/qwen35_4b"
CONDA_ENV="qwen3_5"
BACKEND="sglang"
NUM_PROMPTS=${NUM_PROMPTS:-2048}
N_ROLLOUTS=${N_ROLLOUTS:-2}
MAX_PROMPT=${MAX_PROMPT:-1024}
MAX_COMPLETION=${MAX_COMPLETION:-1024}
SCORE_BATCH=${SCORE_BATCH:-4}
GEN_BATCH_HINT=${GEN_BATCH_HINT:-64}

mkdir -p "${OUTPUT_DIR}" "${BASE_DIR}/log/data_analysis/21"

cd "${BASE_DIR}"
set +u
source activate "qwen3_5"
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -d /usr/local/cuda-12.8 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
elif command -v module >/dev/null 2>&1; then
  module load cuda/12.8 2>/dev/null || true
fi
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

if [[ "${SKIP_GENERATE:-0}" != "1" ]]; then
  echo "[analysis] ===== phase 1: generate ====="
  python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-score
else
  echo "[analysis] ===== phase 1: skipped (SKIP_GENERATE=1; reuse rollouts.jsonl) ====="
fi

echo "[analysis] ===== phase 2: score ====="
python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-generate

echo "[analysis] done -> ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
