#!/bin/bash
# Shared launcher for OPSD data analysis scripts.
# Source or call with environment variables set by per-model wrappers.
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}
export BASE_DIR
ANALYSIS_DIR="${BASE_DIR}/scripts/data_analysis"

# Required env (set by wrapper):
#   TASK          combinations | teacher_prefix | entropy | length_windows
#   MODEL_KEY     registry key, e.g. qwen3_1.7b
#   COMBO         st_tt | snt_tnt | ...
# Optional:
#   ENTROPY_BUCKET   he20 | le20 | he80 | le80
#   CONDA_ENV        override conda env
#   BACKEND          vllm | sglang
#   NUM_PROMPTS      default 2048
#   N_ROLLOUTS       default 2
#   MAX_COMPLETION   default 1024 (6144 for length_windows)
#   SCORE_BATCH      default auto from task+model (short 2–8, length_windows 1–2)
#   GEN_BATCH_HINT   default auto from task+model (SGLang only)
#   JOB_TAG          default SLURM_JOB_ID

: "${TASK:?Set TASK}"
: "${MODEL_KEY:?Set MODEL_KEY}"
: "${COMBO:?Set COMBO}"

# Defaults from registry via python
eval "$(python - <<'PY'
import os, sys
sys.path.insert(0, os.environ["BASE_DIR"] + "/scripts/data_analysis")
from common.model_registry import (
    get_model_config,
    model_launch_overrides,
    task_default_gen_batch_hint,
    task_default_max_completion,
    task_default_score_batch,
)
m = get_model_config(os.environ["MODEL_KEY"])
ov = model_launch_overrides(os.environ["MODEL_KEY"])
task = os.environ["TASK"]
model_key = os.environ["MODEL_KEY"]
print(f'export MODEL_PATH_DEFAULT="{m.model_path}"')
print(f'export CONDA_ENV_DEFAULT="{m.conda_env}"')
print(f'export BACKEND_DEFAULT="{ov.get("backend", m.backend)}"')
print(f'export MAX_COMPLETION_DEFAULT="{task_default_max_completion(task)}"')
print(f'export SCORE_BATCH_DEFAULT="{task_default_score_batch(task, model_key)}"')
print(f'export GEN_BATCH_HINT_DEFAULT="{task_default_gen_batch_hint(task, model_key)}"')
print(f'export MEM_FRACTION_STATIC_DEFAULT="{ov.get("mem_fraction_static", 0.80)}"')
if m.reasoning_parser:
    print(f'export REASONING_PARSER_DEFAULT="{m.reasoning_parser}"')
if ov.get("disable_piecewise_cuda_graph"):
    print('export DISABLE_PIECEWISE_CUDA_GRAPH_DEFAULT=1')
PY
)"

MODEL_PATH=${MODEL_PATH:-${MODEL_PATH_DEFAULT}}
CONDA_ENV=${CONDA_ENV:-${CONDA_ENV_DEFAULT}}
BACKEND=${BACKEND:-${BACKEND_DEFAULT}}
NUM_PROMPTS=${NUM_PROMPTS:-2048}
N_ROLLOUTS=${N_ROLLOUTS:-2}
MAX_PROMPT=${MAX_PROMPT:-1024}
MAX_COMPLETION=${MAX_COMPLETION:-${MAX_COMPLETION_DEFAULT}}
SCORE_BATCH=${SCORE_BATCH:-${SCORE_BATCH_DEFAULT}}
GEN_BATCH_HINT=${GEN_BATCH_HINT:-${GEN_BATCH_HINT_DEFAULT}}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-${MEM_FRACTION_STATIC_DEFAULT:-0.80}}
JOB_TAG=${JOB_TAG:-${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}

RUN_SUFFIX="${COMBO}"
[[ -n "${ENTROPY_BUCKET:-}" ]] && RUN_SUFFIX="${RUN_SUFFIX}_${ENTROPY_BUCKET}"
OUTPUT_DIR=${OUTPUT_DIR:-${BASE_DIR}/scripts/data_analysis/outputs/${TASK}/${MODEL_KEY}/${RUN_SUFFIX}_${JOB_TAG}}
LOG_DIR=${LOG_DIR:-${BASE_DIR}/log/data_analysis/${TASK}/${MODEL_KEY}}
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cd "${BASE_DIR}"
set +u
source activate "${CONDA_ENV}"
set -u

if [[ "${CONDA_ENV}" == "falcon" ]]; then
  # Match scripts/train/falcon_h1r_7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts.sh
  _PY_VER=$(python -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
  _NVIDIA_LIB_ROOT="${CONDA_PREFIX}/lib/${_PY_VER}/site-packages/nvidia"
  _NVIDIA_LD=""
  if [[ -d "${_NVIDIA_LIB_ROOT}/cuda_runtime/lib" ]]; then
    _NVIDIA_LD="${_NVIDIA_LIB_ROOT}/cuda_runtime/lib"
  fi
  if [[ -d "${_NVIDIA_LIB_ROOT}" ]]; then
    for _lib in "${_NVIDIA_LIB_ROOT}"/*/lib; do
      [[ -d "${_lib}" && "${_lib}" != "${_NVIDIA_LIB_ROOT}/cuda_runtime/lib" ]] \
        && _NVIDIA_LD="${_NVIDIA_LD:+${_NVIDIA_LD}:}${_lib}"
    done
  fi
  export LD_LIBRARY_PATH="${_NVIDIA_LD:+${_NVIDIA_LD}:}${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  if [[ ! -e "${CONDA_PREFIX}/lib64/libcudart.so" && -f "${CONDA_PREFIX}/targets/x86_64-linux/lib/libcudart.so" ]]; then
    mkdir -p "${CONDA_PREFIX}/lib64"
    ln -sf "${CONDA_PREFIX}/targets/x86_64-linux/lib/libcudart.so" "${CONDA_PREFIX}/lib64/libcudart.so"
  fi
  if [[ -d "${_NVIDIA_LIB_ROOT}/cuda_runtime/lib" ]]; then
    export LIBRARY_PATH="${_NVIDIA_LIB_ROOT}/cuda_runtime/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  fi
  if command -v module >/dev/null 2>&1; then
    module load gcc/11 2>/dev/null || module load gcc/9 2>/dev/null || true
  fi
  if [[ -n "${_NVIDIA_LD}" ]]; then
    export LD_LIBRARY_PATH="${_NVIDIA_LD}:${LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${_NVIDIA_LIB_ROOT}/cuda_runtime/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  fi
  unset PYTORCH_CUDA_ALLOC_CONF
  export SGLANG_MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}"
  export SGLANG_ATTENTION_BACKEND=triton
  export SGLANG_SAMPLING_BACKEND=pytorch
  export SGLANG_REASONING_PARSER="${REASONING_PARSER:-${REASONING_PARSER_DEFAULT:-deepseek-r1}}"
  export SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH="${DISABLE_PIECEWISE_CUDA_GRAPH:-${DISABLE_PIECEWISE_CUDA_GRAPH_DEFAULT:-1}}"
  echo "[analysis] conda=falcon conda_prefix=${CONDA_PREFIX} py_ver=${_PY_VER}"
  echo "[analysis] LD_head=$(echo "${LD_LIBRARY_PATH}" | cut -d: -f1-3)"
  python -c "import mamba_ssm, causal_conv1d" >/dev/null
elif [[ "${CONDA_ENV}" == "qwen3_5" ]]; then
  if [[ -d /usr/local/cuda-12.8 ]]; then
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
  elif command -v module >/dev/null 2>&1; then
    module load cuda/12.8 2>/dev/null || true
  fi
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export PYTHONPATH="${BASE_DIR}/src:${BASE_DIR}/scripts/data_analysis:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_CONFIGURE_LOGGING=0

EXTRA_ARGS=()
EXTRA_ARGS+=(--task "${TASK}")
EXTRA_ARGS+=(--model-key "${MODEL_KEY}")
EXTRA_ARGS+=(--combo "${COMBO}")
EXTRA_ARGS+=(--model-path "${MODEL_PATH}")
EXTRA_ARGS+=(--output-dir "${OUTPUT_DIR}")
EXTRA_ARGS+=(--num-prompts "${NUM_PROMPTS}")
EXTRA_ARGS+=(--n-rollouts "${N_ROLLOUTS}")
EXTRA_ARGS+=(--max-prompt-length "${MAX_PROMPT}")
EXTRA_ARGS+=(--max-completion-length "${MAX_COMPLETION}")
EXTRA_ARGS+=(--temperature 1.1)
EXTRA_ARGS+=(--top-p 0.95)
EXTRA_ARGS+=(--top-k 20)
EXTRA_ARGS+=(--score-batch-size "${SCORE_BATCH}")
EXTRA_ARGS+=(--gen-batch-hint "${GEN_BATCH_HINT}")
EXTRA_ARGS+=(--backend "${BACKEND}")
EXTRA_ARGS+=(--gpu-memory-utilization 0.90)
EXTRA_ARGS+=(--seed 42)

if [[ -n "${ENTROPY_BUCKET:-}" ]]; then
  EXTRA_ARGS+=(--entropy-bucket "${ENTROPY_BUCKET}")
fi
if [[ -n "${REASONING_PARSER:-${REASONING_PARSER_DEFAULT:-}}" ]]; then
  EXTRA_ARGS+=(--reasoning-parser "${REASONING_PARSER:-${REASONING_PARSER_DEFAULT}}")
fi
if [[ "${BACKEND}" == "sglang" ]]; then
  EXTRA_ARGS+=(--attention-backend triton --sampling-backend pytorch --mem-fraction-static "${MEM_FRACTION_STATIC}")
  if [[ "${DISABLE_PIECEWISE_CUDA_GRAPH:-${DISABLE_PIECEWISE_CUDA_GRAPH_DEFAULT:-0}}" == "1" ]]; then
    EXTRA_ARGS+=(--disable-piecewise-cuda-graph)
  fi
fi

COMMON_ARGS=("${EXTRA_ARGS[@]}")

echo "[analysis] task=${TASK} model=${MODEL_KEY} combo=${COMBO}"
echo "[analysis] model_path=${MODEL_PATH} backend=${BACKEND} conda=${CONDA_ENV}"
echo "[analysis] output=${OUTPUT_DIR}"
echo "[analysis] prompts=${NUM_PROMPTS} rollouts=${N_ROLLOUTS} max_completion=${MAX_COMPLETION}"
echo "[analysis] score_batch=${SCORE_BATCH} gen_batch_hint=${GEN_BATCH_HINT}"

_run_phase() {
  local flag=$1
  python "${ANALYSIS_DIR}/run_opsd_analysis.py" "${COMMON_ARGS[@]}" "${flag}"
}

echo "[analysis] ===== phase 1: generate ====="
_run_phase --skip-score

echo "[analysis] ===== phase 2: score ====="
_run_phase --skip-generate

echo "[analysis] done -> ${OUTPUT_DIR}"
ls -lah "${OUTPUT_DIR}"
