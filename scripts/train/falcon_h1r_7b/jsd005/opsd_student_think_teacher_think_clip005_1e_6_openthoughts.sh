#!/bin/bash
#SBATCH --job-name=st_tt_clip005_1e6_falcon7b
#SBATCH --output=log/train/falcon_h1r_7b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=72:00:00
set -euo pipefail

# Falcon-H1R-7B OPSD (falcon version of mimo_7b_rl jsd005 st_tt clip005 1e-6).
# Aligned with scripts/eval/falcon_h1r_7b/*_sgl.sh:
#   model=/.../falcon-h1r-7b, conda=falcon, rollout=SGLang (triton + pytorch sampling),
#   reasoning_parser=deepseek-r1, disable_piecewise_cuda_graph (Mamba Dynamo workaround).
# Chat template has no enable_thinking switch; system prompt already asks for
# <think> blocks (match eval --enable-thinking, which is a no-op in template).
# jsd_token_clip=0.05, lr=1e-6.
# Batch: micro=2, gas=4, 4 GPU → global_batch=32 (same samples/update as mimo7b baseline).
#
# Env-overridable for longer/shorter student rollouts while keeping samples/update fixed:
#   MAX_COMPLETION_LENGTH / PER_DEVICE_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS
#   TARGET_GLOBAL_BATCH (default 32) is asserted: micro * gas * num_gpus == target.
#
# Examples:
#   MAX_STEPS=2 sbatch this.sh
#   MAX_COMPLETION_LENGTH=256 sbatch this.sh   # or use jsd005/length/opsd_st_tt_clip005_c256_1e6_ot.sh

MODE=${MODE:-opsd}
TEACHER_PRIVILEGE_FIELD=${TEACHER_PRIVILEGE_FIELD:-solution}
# Falcon chat template ignores enable_thinking; match eval (freeform CoT in template).
STUDENT_THINKING=${STUDENT_THINKING:-0}
TEACHER_THINKING=${TEACHER_THINKING:-0}

LEARNING_RATE=${LEARNING_RATE:-1e-6}
JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-1024}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH:-32}
MAX_STEPS=${MAX_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-25}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.40}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-triton}
SGLANG_SAMPLING_BACKEND=${SGLANG_SAMPLING_BACKEND:-pytorch}
SGLANG_REASONING_PARSER=${SGLANG_REASONING_PARSER:-deepseek-r1}
SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH=${SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH:-1}
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-sglang}

RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_openthoughts_falcon7b}

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/falcon-h1r-7b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.opsd.solution.nothink.falconh1r7b.maxprompt1024.parquet}
: "${DATASET_PATH:?Set DATASET_PATH to the preprocessed OpenThoughts parquet path}"
MODEL_TAG=${MODEL_TAG:-falcon_h1r_7b}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/${MODEL_TAG}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/${JOB_TAG}}
RUN_NAME_WITH_JOB=${RUN_NAME}_${JOB_TAG}

cd "${BASE_DIR}"
set +u
source activate falcon
set -u
# Match eval falcon_h1r_7b: minimal LD (conda lib only). For SGLang spawn children,
# prepend nvidia pip cudart (cudaGetDriverEntryPointByVersion) — do NOT append
# /usr/local/cuda/lib64; that old libcudart breaks torch in scheduler subprocesses.
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
# SGLang JIT links with `-L$CONDA/lib64 -lcudart`; ensure lib64 has libcudart.so for the linker.
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
# Re-assert nvidia cudart after module load (gcc prepends its lib64).
if [[ -n "${_NVIDIA_LD}" ]]; then
  export LD_LIBRARY_PATH="${_NVIDIA_LD}:${LD_LIBRARY_PATH}"
  export LIBRARY_PATH="${_NVIDIA_LIB_ROOT}/cuda_runtime/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi

export PYTHONPATH="${BASE_DIR}/src:${BASE_DIR}/vendor/verl:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export WANDB_MODE=offline
export WANDB_PROJECT=${WANDB_PROJECT:-OPSD}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-falcon_h1r_7b_fullparam_100step_openthoughts}
export WANDB_DIR=${WANDB_DIR:-${BASE_DIR}/wandb}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HYDRA_FULL_ERROR=1
export ROLLOUT_BACKEND
export SGLANG_MEM_FRACTION_STATIC
export SGLANG_ATTENTION_BACKEND
export SGLANG_SAMPLING_BACKEND
export SGLANG_REASONING_PARSER
export SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH
unset PYTORCH_CUDA_ALLOC_CONF
echo "[launch] conda=falcon conda_prefix=${CONDA_PREFIX} py_ver=${_PY_VER}"
echo "[launch] LD_head=$(echo "${LD_LIBRARY_PATH}" | cut -d: -f1-3)"

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_HOME}" \
  "${BASE_DIR}/log/train/falcon_h1r_7b"

python -c "import mamba_ssm, causal_conv1d" >/dev/null

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing preprocessed dataset: ${DATASET_PATH}" >&2
  echo "[error] run: sbatch scripts/data/preprocess_opsd_openthoughts_falcon_h1r_7b_nothink.sh" >&2
  exit 1
fi

# Match eval falcon_h1r_7b (no enable_thinking in chat template).
THINK_ARGS=(--no-student-thinking --no-teacher-thinking)
MASTER_PORT=${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}

NUM_GPUS=${NUM_GPUS:-4}
GLOBAL_BATCH=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))
TOTAL_SAMPLES=$((GLOBAL_BATCH * MAX_STEPS))
MAX_SEQ_LEN=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))

if [[ "${GLOBAL_BATCH}" -ne "${TARGET_GLOBAL_BATCH}" ]]; then
  echo "[error] global_batch=${GLOBAL_BATCH} != TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH}" >&2
  echo "[error] keep samples/update fixed: micro * gas * gpus must equal ${TARGET_GLOBAL_BATCH}" >&2
  exit 1
fi

if [[ "${JSD_TOKEN_CLIP}" == "none" || "${JSD_TOKEN_CLIP}" == "None" || "${JSD_TOKEN_CLIP}" == "NONE" ]]; then
  JSD_TOKEN_CLIP=0
fi

SGLANG_EXTRA_ARGS=()
if [[ -n "${SGLANG_REASONING_PARSER}" ]]; then
  SGLANG_EXTRA_ARGS+=(--sglang-reasoning-parser "${SGLANG_REASONING_PARSER}")
fi
if [[ "${SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH}" == "1" ]]; then
  SGLANG_EXTRA_ARGS+=(--sglang-disable-piecewise-cuda-graph)
fi

echo "[launch] run=${RUN_NAME_WITH_JOB} mode=${MODE} privilege_field=${TEACHER_PRIVILEGE_FIELD}"
echo "[launch] student_thinking=${STUDENT_THINKING} teacher_thinking=${TEACHER_THINKING} (Falcon: template CoT; match eval)"
echo "[launch] lr=${LEARNING_RATE} jsd_token_clip=${JSD_TOKEN_CLIP}"
echo "[launch] micro=${PER_DEVICE_BATCH_SIZE} gas=${GRADIENT_ACCUMULATION_STEPS} gpus=${NUM_GPUS} → global_batch=${GLOBAL_BATCH}"
echo "[launch] max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS} → total_samples=${TOTAL_SAMPLES}"
echo "[launch] prompt=${MAX_PROMPT_LENGTH} completion=${MAX_COMPLETION_LENGTH} max_seq=${MAX_SEQ_LEN}"
echo "[launch] model=${MODEL_PATH} dataset=${DATASET_PATH} output=${OUTPUT_DIR}"
echo "[launch] master_port=${MASTER_PORT} rollout=${ROLLOUT_BACKEND} sglang_mem=${SGLANG_MEM_FRACTION_STATIC} attn=${SGLANG_ATTENTION_BACKEND} sampling=${SGLANG_SAMPLING_BACKEND}"
echo "[launch] reasoning_parser=${SGLANG_REASONING_PARSER:-none} disable_piecewise_cuda_graph=${SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH}"

accelerate launch \
  --config_file "${BASE_DIR}/configs/accelerate_zero3.yaml" \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MASTER_PORT}" \
  "${BASE_DIR}/src/train_opsd.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME_WITH_JOB}" \
  --privilege-mode "${MODE}" \
  --teacher-privilege-field "${TEACHER_PRIVILEGE_FIELD}" \
  --max-steps "${MAX_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --max-prompt-length "${MAX_PROMPT_LENGTH}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --jsd-token-clip "${JSD_TOKEN_CLIP}" \
  --rollout-backend "${ROLLOUT_BACKEND}" \
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
  --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}" \
  --sglang-sampling-backend "${SGLANG_SAMPLING_BACKEND}" \
  --deepspeed "${BASE_DIR}/configs/deepspeed_zero3.json" \
  "${SGLANG_EXTRA_ARGS[@]}" \
  "${THINK_ARGS[@]}"
