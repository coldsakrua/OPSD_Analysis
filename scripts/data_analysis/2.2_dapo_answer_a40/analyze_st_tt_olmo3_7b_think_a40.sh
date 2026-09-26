#!/bin/bash
#SBATCH --job-name=da22dapo_a40_st_tt_olmo7bt
#SBATCH --output=log/data_analysis/22_dapo/%x.%j.out
#SBATCH --partition=GPUA40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=54G
#SBATCH --time=96:00:00
set -euo pipefail
# DAPO answer-only teacher_prefix (st_tt / olmo3_7b_think)
# Shared 2048 problems × 2 rollouts; privilege=answer (correct+GT).
# Seq ≤2048; SCORE_BATCH small/med/large=32/16/8; vLLM util=0.95; SGLang mem=0.80 (falcon 0.55).
# Dataset: same DAPO-Math GT pool as qwen3-1.7b/4b dapomath train.
BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
RUN_SUFFIX="dapo_answer"
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/scripts/data_analysis/outputs/teacher_prefix_dapo/olmo3_7b_think/${RUN_SUFFIX}_${JOB_TAG}}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/dapo/preprocessed/dapo-math-17k.answer_only.shared2048.seed42.maxprompt1024.parquet}
TASK="teacher_prefix"
MODEL_KEY="olmo3_7b_think"
COMBO="st_tt"
MODEL_PATH="/gpfs/share/home/2501210611/labShare/2501210611/model/olmo-3-7b-think"
CONDA_ENV="sglang"
BACKEND="sglang"
NUM_PROMPTS=${NUM_PROMPTS:-2048}
N_ROLLOUTS=${N_ROLLOUTS:-2}
MAX_PROMPT=${MAX_PROMPT:-1024}
MAX_COMPLETION=${MAX_COMPLETION:-1024}
SCORE_BATCH=${SCORE_BATCH:-4}
GEN_BATCH_HINT=${GEN_BATCH_HINT:-32}
mkdir -p "${OUTPUT_DIR}" "${BASE_DIR}/log/data_analysis/22_dapo"
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing shared DAPO parquet: ${DATASET_PATH}" >&2
  echo "[error] run: sbatch scripts/data/prepare_dapo_answer_shared2048.sh" >&2
  exit 1
fi
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
export SGLANG_MEM_FRACTION_STATIC="0.75"
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
  --dataset-path "${DATASET_PATH}"
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
  --gpu-memory-utilization 0.95
  --seed 42
  --teacher-prefixes answer
  --no-save-token-metrics
)
EXTRA_ARGS+=(--attention-backend triton --sampling-backend pytorch --mem-fraction-static 0.75)
echo "[analysis] task=${TASK} model=${MODEL_KEY} combo=${COMBO} prefix=answer backend=${BACKEND}"
echo "[analysis] dataset=${DATASET_PATH}"
echo "[analysis] output=${OUTPUT_DIR}"
echo "[analysis] score_batch=${SCORE_BATCH} gen_batch_hint=${GEN_BATCH_HINT}"
if [[ "${SKIP_GENERATE:-0}" != "1" ]]; then
  echo "[analysis] ===== phase 1: generate ====="
  python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-score
else
  echo "[analysis] ===== phase 1: skipped (SKIP_GENERATE=1; reuse rollouts.jsonl) ====="
fi
echo "[analysis] ===== phase 2: score answer ====="
python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-generate
echo "[analysis] done -> ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
