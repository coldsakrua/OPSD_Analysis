#!/bin/bash
#SBATCH --exclude=gpua800n13,gpua800n21
#SBATCH --job-name=da26_st_tt_olmo7bt_hard
#SBATCH --output=log/data_analysis/26/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
set -euo pipefail

# 2.6 cotlen-hard st_tt on olmo3_7b_think
# OT cotlen band=hard: prompt=2048 completion=32768; n=4 rollouts.
# Default: generate + boxed accuracy only. Preference/data-analysis: RUN_PREFERENCE=1 (or submit_preference.sh).
# Rollout: temp=1.1 top_p=0.95 top_k=20 max_prompt=2048 max_completion=32768

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/scripts/data_analysis/outputs/cotlen/olmo3_7b_think/st_tt_hard_${JOB_TAG}}

TASK="cotlen"
MODEL_KEY="olmo3_7b_think"
COMBO="st_tt"
MODEL_PATH="/gpfs/share/home/2501210611/labShare/2501210611/model/olmo-3-7b-think"
CONDA_ENV="sglang"
BACKEND="sglang"
NUM_PROMPTS=${NUM_PROMPTS:-2048}
N_ROLLOUTS=${N_ROLLOUTS:-4}
MAX_PROMPT=${MAX_PROMPT:-2048}
MAX_COMPLETION=${MAX_COMPLETION:-32768}
SCORE_BATCH=${SCORE_BATCH:-1}
GEN_BATCH_HINT=${GEN_BATCH_HINT:-2}

mkdir -p "${OUTPUT_DIR}" "${BASE_DIR}/log/data_analysis/26"

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
EXTRA_ARGS+=(--cotlen-band "hard")
EXTRA_ARGS+=(--attention-backend triton --sampling-backend pytorch --mem-fraction-static 0.8)

echo "[analysis] task=${TASK} model=${MODEL_KEY} combo=${COMBO} backend=${BACKEND}"
echo "[analysis] output=${OUTPUT_DIR}"

if [[ "${SKIP_GENERATE:-0}" != "1" ]]; then
  echo "[analysis] ===== phase A: generate + accuracy ====="
  python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-score
elif [[ "${RUN_ACCURACY:-1}" == "1" ]]; then
  echo "[analysis] ===== phase A: accuracy only (reuse rollouts.jsonl) ====="
  python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-generate --skip-score
else
  echo "[analysis] ===== phase A: skipped ====="
fi

if [[ "${RUN_PREFERENCE:-0}" == "1" ]]; then
  echo "[analysis] ===== phase B: preference / data analysis ====="
  python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-generate --skip-accuracy
else
  echo "[analysis] ===== phase B: preference skipped (set RUN_PREFERENCE=1) ====="
fi

echo "[analysis] done -> ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
