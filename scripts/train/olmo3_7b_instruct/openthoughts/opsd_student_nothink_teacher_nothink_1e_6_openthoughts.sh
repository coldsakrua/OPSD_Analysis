#!/bin/bash
#SBATCH --job-name=snt_tnt_olmo7bi
#SBATCH --output=log/train/olmo3-7b-instruct/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=03:00:00
# Exclude n13/n21: known bad/old CUDA runtime that breaks torch+cu126 (same as qwen3.5 scripts).
#SBATCH --exclude=gpua800n03,gpua800n10,gpua800n13,gpua800n21
set -euo pipefail

# Olmo-3-7B-Instruct OPSD (student/teacher both no-think, teacher gets full solution).
# Hyperparams aligned with qwen3_4b_instruct OpenThoughts baseline:
#   LR=1e-6, JSD_TOKEN_CLIP=1e-6, micro=4, gas=4, 4 GPU → global_batch=64
# Rollout: SGLang Engine (triton), not vLLM.
#
# Examples:
#   MAX_STEPS=2 sbatch this.sh
#   PER_DEVICE_BATCH_SIZE=2 SGLANG_MEM_FRACTION_STATIC=0.35 sbatch this.sh

MODE=${MODE:-opsd}
TEACHER_PRIVILEGE_FIELD=${TEACHER_PRIVILEGE_FIELD:-solution}
STUDENT_THINKING=${STUDENT_THINKING:-0}
TEACHER_THINKING=${TEACHER_THINKING:-0}

LEARNING_RATE=${LEARNING_RATE:-1e-6}
JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-1e-6}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
MAX_STEPS=${MAX_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-25}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.40}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-triton}
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-sglang}

RUN_NAME=${RUN_NAME:-snt_tnt_1e_6_openthoughts_olmo7bit}

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/olmo3-7b-it}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.opsd.solution.nothink.olmo7bit.maxprompt1024.parquet}
: "${DATASET_PATH:?Set DATASET_PATH to the preprocessed OpenThoughts parquet path}"
MODEL_TAG=${MODEL_TAG:-olmo3_7b_instruct}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/${MODEL_TAG}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/${JOB_TAG}}
RUN_NAME_WITH_JOB=${RUN_NAME}_${JOB_TAG}

cd "${BASE_DIR}"
set +u
source activate sglang
set -u
# Torch 2.8+cu126 needs nvidia pip cudart (cudaGetDriverEntryPointByVersion).
# Putting only CONDA_PREFIX/lib first lets node /usr/local/cuda/lib64 (older) win via LD_LIBRARY_PATH
# over torch RUNPATH → ImportError in SGLang spawn children.
_NVIDIA_LIB_ROOT="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia"
_NVIDIA_LD=""
if [[ -d "${_NVIDIA_LIB_ROOT}" ]]; then
  for _lib in "${_NVIDIA_LIB_ROOT}"/*/lib; do
    [[ -d "${_lib}" ]] && _NVIDIA_LD="${_NVIDIA_LD:+${_NVIDIA_LD}:}${_lib}"
  done
fi
export LD_LIBRARY_PATH="${_NVIDIA_LD:+${_NVIDIA_LD}:}${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Prefer a CUDA toolkit matching torch (cu126) when present on the compute node (for nvcc/JIT).
# Keep nvidia pip libs first — do not let toolkit lib64 shadow cudaGetDriverEntryPointByVersion.
if [[ -d /usr/local/cuda-12.6 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.6
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
elif [[ -d /usr/local/cuda-12.8 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"
elif command -v module >/dev/null 2>&1; then
  module load cuda/12.6 2>/dev/null || module load cuda/12.8 2>/dev/null || true
fi
# Re-assert nvidia libs at the front after any module load that may prepend system CUDA.
if [[ -n "${_NVIDIA_LD}" ]]; then
  export LD_LIBRARY_PATH="${_NVIDIA_LD}:${LD_LIBRARY_PATH}"
fi
# Prefer newer GCC for optional torch JIT extensions (c_dlpack); system GCC 8 is too old.
if command -v module >/dev/null 2>&1; then
  module load gcc/11 2>/dev/null || module load gcc/9 2>/dev/null || true
fi

# #region agent log
_DBG_LOG="/gpfs/share/home/2501210611/opsd_analysis/.cursor/debug-470747.log"
mkdir -p "$(dirname "${_DBG_LOG}")"
python - <<'PY' >>"${_DBG_LOG}" 2>/dev/null || true
import json, os, socket, subprocess, time
from pathlib import Path
payload = {
    "sessionId": "470747",
    "runId": "post-fix",
    "hypothesisId": "A",
    "location": "opsd_student_nothink_teacher_nothink_1e_6_openthoughts.sh:cuda_env",
    "message": "olmo train CUDA env at launch",
    "timestamp": int(time.time() * 1000),
    "data": {
        "hostname": socket.gethostname(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "ld_library_path_head": (os.environ.get("LD_LIBRARY_PATH") or "")[:500],
        "nvidia_cudart": str(Path(os.environ.get("CONDA_PREFIX","")) / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"),
        "nvidia_cudart_exists": (Path(os.environ.get("CONDA_PREFIX","")) / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12").exists(),
        "usr_local_cuda_exists": Path("/usr/local/cuda").exists(),
        "usr_local_cuda126_exists": Path("/usr/local/cuda-12.6").exists(),
        "usr_local_cuda128_exists": Path("/usr/local/cuda-12.8").exists(),
        "gcc": subprocess.getoutput("gcc --version").splitlines()[:1],
    },
}
# resolve libcudart via a tiny probe
try:
    import ctypes
    lib = ctypes.CDLL("libcudart.so.12")
    payload["data"]["cudart_load"] = "ok"
    payload["data"]["has_cudaGetDriverEntryPointByVersion"] = hasattr(lib, "cudaGetDriverEntryPointByVersion")
except Exception as e:
    payload["data"]["cudart_load"] = f"fail:{type(e).__name__}:{e}"
print(json.dumps(payload, ensure_ascii=False))
PY
# #endregion

export PYTHONPATH="${BASE_DIR}/src:${BASE_DIR}/vendor/verl:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export WANDB_MODE=offline
export WANDB_PROJECT=${WANDB_PROJECT:-OPSD}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-olmo3_7b_instruct_fullparam_100step_openthoughts}
export WANDB_DIR=${WANDB_DIR:-${BASE_DIR}/wandb}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HYDRA_FULL_ERROR=1
export ROLLOUT_BACKEND
export SGLANG_MEM_FRACTION_STATIC
export SGLANG_ATTENTION_BACKEND
unset PYTORCH_CUDA_ALLOC_CONF
echo "[launch] CUDA_HOME=${CUDA_HOME:-unset} LD_head=$(echo "${LD_LIBRARY_PATH}" | cut -d: -f1-3)"

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_HOME}" \
  "${BASE_DIR}/log/train/olmo3-7b-instruct"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing preprocessed dataset: ${DATASET_PATH}" >&2
  echo "[error] run: PYTHONPATH=src:vendor/verl python scripts/data/preprocess_opsd_openthoughts.py --privilege-mode opsd --teacher-privilege-field solution --no-student-thinking --no-teacher-thinking --model-path ${MODEL_PATH} --output data/openthoughts/preprocessed/openthoughts.opsd.solution.nothink.olmo7bit.maxprompt1024.parquet" >&2
  exit 1
fi

THINK_ARGS=(--no-student-thinking --no-teacher-thinking)
MASTER_PORT=${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}

NUM_GPUS=4
GLOBAL_BATCH=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))

if [[ "${JSD_TOKEN_CLIP}" == "none" || "${JSD_TOKEN_CLIP}" == "None" || "${JSD_TOKEN_CLIP}" == "NONE" ]]; then
  JSD_TOKEN_CLIP=0
fi

echo "[launch] run=${RUN_NAME_WITH_JOB} mode=${MODE} privilege_field=${TEACHER_PRIVILEGE_FIELD}"
echo "[launch] student_thinking=${STUDENT_THINKING} teacher_thinking=${TEACHER_THINKING}"
echo "[launch] lr=${LEARNING_RATE} jsd_token_clip=${JSD_TOKEN_CLIP}"
echo "[launch] micro=${PER_DEVICE_BATCH_SIZE} gas=${GRADIENT_ACCUMULATION_STEPS} gpus=${NUM_GPUS} → global_batch=${GLOBAL_BATCH}"
echo "[launch] max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS}"
echo "[launch] model=${MODEL_PATH} dataset=${DATASET_PATH} output=${OUTPUT_DIR}"
echo "[launch] master_port=${MASTER_PORT} rollout=${ROLLOUT_BACKEND} sglang_mem=${SGLANG_MEM_FRACTION_STATIC} attn=${SGLANG_ATTENTION_BACKEND}"

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
  --max-prompt-length 1024 \
  --max-completion-length 1024 \
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --jsd-token-clip "${JSD_TOKEN_CLIP}" \
  --rollout-backend "${ROLLOUT_BACKEND}" \
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
  --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}" \
  --deepspeed "${BASE_DIR}/configs/deepspeed_zero3.json" \
  "${THINK_ARGS[@]}"
