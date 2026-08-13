#!/bin/bash
#SBATCH --job-name=st_tt_lora_q35
#SBATCH --output=log/train/qwen3.5_4b/lora/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=3:00:00
#SBATCH --exclude=gpua800n13
set -euo pipefail

# LoRA OPSD on Qwen3.5-4B with student+teacher both think.
# LoRA hyperparams match official siyan-zhao/OPSD run_opsd_1b.sh:
#   use_peft + fixed_teacher, lora_r=64, lora_alpha=128
#   lr=5e-6, jsd_token_clip=0.05, max_completion=1024
#   2×A800: micro=4, gas=4 → global_batch=32
#   ZeRO-2 without CPU optimizer offload:
#     - CPU offload needs DeepSpeedCPUAdam JIT (fails on GCC 8.5 nodes)
#     - ZeRO-3 + LoRA + gradient checkpointing hits CheckpointError on Qwen3.5
#
# Rollout: SGLang (triton) in conda env qwen3_5; sampling matches
# scripts/eval/qwen3.5_4b/*_think.sh (temp=1.0 top_p=0.95 top_k=20 presence_penalty=1.5).

MODE=${MODE:-opsd}
TEACHER_PRIVILEGE_FIELD=${TEACHER_PRIVILEGE_FIELD:-solution}
STUDENT_THINKING=${STUDENT_THINKING:-1}
TEACHER_THINKING=${TEACHER_THINKING:-1}

LEARNING_RATE=${LEARNING_RATE:-5e-6}
JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-1024}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH:-32}
MAX_STEPS=${MAX_STEPS:-100}
SAVE_STEPS=${SAVE_STEPS:-25}
LORA_R=${LORA_R:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
# Eval-aligned rollout sampling (qwen3.5 think)
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-1.5}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.40}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-triton}
SGLANG_SAMPLING_BACKEND=${SGLANG_SAMPLING_BACKEND:-pytorch}
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-sglang}

RUN_NAME=${RUN_NAME:-st_tt_lora_clip005_lr5e6_openthoughts_q35_4b}

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}}
MODEL_PATH=${MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/qwen35_4b}
DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.opsd.solution.sthink_tthink.qwen35_4b.maxprompt1024.parquet}
: "${DATASET_PATH:?Set DATASET_PATH to the preprocessed OpenThoughts parquet path}"
MODEL_TAG=${MODEL_TAG:-qwen3.5_4b}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BASE_DIR}/outputs/${MODEL_TAG}}
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}/${JOB_TAG}}
RUN_NAME_WITH_JOB=${RUN_NAME}_${JOB_TAG}

cd "${BASE_DIR}"
set +u
source activate qwen3_5
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -d /usr/local/cuda-12.8 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
elif command -v module >/dev/null 2>&1; then
  module load cuda/12.8 2>/dev/null || true
fi

export PYTHONPATH="${BASE_DIR}/src:${BASE_DIR}/vendor/verl:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export WANDB_MODE=offline
export WANDB_PROJECT=${WANDB_PROJECT:-OPSD}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen35_4b_lora_official_aligned}
export WANDB_DIR=${WANDB_DIR:-${BASE_DIR}/wandb_runs}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export HYDRA_FULL_ERROR=1
export ROLLOUT_BACKEND
export SGLANG_MEM_FRACTION_STATIC
export SGLANG_ATTENTION_BACKEND
export SGLANG_SAMPLING_BACKEND
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_HOME}" "${BASE_DIR}/log/train/qwen3.5_4b/lora"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing preprocessed dataset: ${DATASET_PATH}" >&2
  echo "[error] run: PYTHONPATH=src:vendor/verl python scripts/data/preprocess_opsd_openthoughts.py --privilege-mode opsd --teacher-privilege-field solution --student-thinking --teacher-thinking --model-path ${MODEL_PATH} --output ${DATASET_PATH}" >&2
  exit 1
fi

THINK_ARGS=(--student-thinking --teacher-thinking)
MASTER_PORT=${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}

NUM_GPUS=2
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

echo "[launch] run=${RUN_NAME_WITH_JOB} mode=${MODE} privilege_field=${TEACHER_PRIVILEGE_FIELD}"
echo "[launch] student_thinking=${STUDENT_THINKING} teacher_thinking=${TEACHER_THINKING}"
echo "[launch] LoRA r=${LORA_R} alpha=${LORA_ALPHA} fixed_teacher=1 (official PEFT path)"
echo "[launch] lr=${LEARNING_RATE} jsd_token_clip=${JSD_TOKEN_CLIP}"
echo "[launch] micro=${PER_DEVICE_BATCH_SIZE} gas=${GRADIENT_ACCUMULATION_STEPS} gpus=${NUM_GPUS} → global_batch=${GLOBAL_BATCH}"
echo "[launch] max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS} → total_samples=${TOTAL_SAMPLES}"
echo "[launch] prompt=${MAX_PROMPT_LENGTH} completion=${MAX_COMPLETION_LENGTH} max_seq=${MAX_SEQ_LEN}"
echo "[launch] sampling temp=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K} min_p=${MIN_P} presence_penalty=${PRESENCE_PENALTY}"
echo "[launch] model=${MODEL_PATH} dataset=${DATASET_PATH} output=${OUTPUT_DIR}"
echo "[launch] accelerate=zero2 deepspeed=zero2_no_offload rollout=${ROLLOUT_BACKEND} sglang_mem=${SGLANG_MEM_FRACTION_STATIC}"
echo "[launch] master_port=${MASTER_PORT}"

accelerate launch \
  --config_file "${BASE_DIR}/configs/accelerate_zero2_no_offload.yaml" \
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
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --min-p "${MIN_P}" \
  --presence-penalty "${PRESENCE_PENALTY}" \
  --use-peft \
  --fixed-teacher \
  --lora-r "${LORA_R}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --rollout-backend "${ROLLOUT_BACKEND}" \
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
  --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}" \
  --sglang-sampling-backend "${SGLANG_SAMPLING_BACKEND}" \
  --deepspeed "${BASE_DIR}/configs/deepspeed_zero2_no_offload.json" \
  "${THINK_ARGS[@]}"
