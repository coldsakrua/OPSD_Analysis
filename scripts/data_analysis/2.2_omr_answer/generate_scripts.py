#!/usr/bin/env python3
"""Generate SLURM scripts for OMR answer-only teacher-prefix analysis (10 models)."""

from __future__ import annotations

import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR.parent))

from common.model_registry import get_model_config, model_launch_overrides  # noqa: E402

# Answer-only at max_prompt=1024 (seq≤2048): push score batch higher.
#   small ≤1.7B → B=48; medium 4B → B=24; large 7B → B=12
SCORE_BATCH = {"small": 48, "medium": 24, "large": 12}
GEN_BATCH = {"small": 256, "medium": 256, "large": 128}
TIER = {
    "qwen3_06b": "small",
    "deepseek_r1_1.5b": "small",
    "qwen3_1.7b": "small",
    "qwen3_4b": "medium",
    "qwen3_4b_instruct": "medium",
    "qwen3_4b_thinking": "medium",
    "olmo3_7b_instruct": "large",
    "olmo3_7b_think": "large",
    "falcon_h1r_7b": "large",
    "mimo_7b_rl": "large",
}

MODELS: list[tuple[str, str, str]] = [
    ("qwen3_06b", "st_tt", "0p6b"),
    ("qwen3_1.7b", "st_tt", "1p7b"),
    ("qwen3_4b", "st_tt", "4b"),
    ("qwen3_4b_instruct", "snt_tnt", "4bi"),
    ("qwen3_4b_thinking", "st_tt", "4bt"),
    ("mimo_7b_rl", "st_tt", "mimo7b"),
    ("deepseek_r1_1.5b", "st_tt", "ds1p5b"),
    ("falcon_h1r_7b", "st_tt", "falcon7b"),
    ("olmo3_7b_instruct", "snt_tnt", "olmo7bi"),
    ("olmo3_7b_think", "st_tt", "olmo7bt"),
]

SBATCH_EXCLUDE = {
    "olmo3_7b_instruct",
    "olmo3_7b_think",
    "falcon_h1r_7b",
    "mimo_7b_rl",
}

SHARED_DS = (
    "${BASE_DIR}/data/openmathreasoning/preprocessed/"
    "omr.correct.answer.shared2048.seed42.maxprompt1024.parquet"
)


def _conda_block(model_key: str, conda_env: str, backend: str, mem_fraction: float, reasoning_parser: str, disable_cuda_graph: bool) -> str:
    lines = [
        'cd "${BASE_DIR}"',
        "set +u",
        f'source activate "{conda_env}"',
        "set -u",
    ]
    if conda_env == "falcon":
        lines.extend(
            [
                '_PY_VER=$(python -c \'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")\')',
                '_NVIDIA_LIB_ROOT="${CONDA_PREFIX}/lib/${_PY_VER}/site-packages/nvidia"',
                '_NVIDIA_LD=""',
                'if [[ -d "${_NVIDIA_LIB_ROOT}/cuda_runtime/lib" ]]; then _NVIDIA_LD="${_NVIDIA_LIB_ROOT}/cuda_runtime/lib"; fi',
                'if [[ -d "${_NVIDIA_LIB_ROOT}" ]]; then',
                '  for _lib in "${_NVIDIA_LIB_ROOT}"/*/lib; do',
                '    [[ -d "${_lib}" && "${_lib}" != "${_NVIDIA_LIB_ROOT}/cuda_runtime/lib" ]] && _NVIDIA_LD="${_NVIDIA_LD:+${_NVIDIA_LD}:}${_lib}"',
                "  done",
                "fi",
                'export LD_LIBRARY_PATH="${_NVIDIA_LD:+${_NVIDIA_LD}:}${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"',
                'if [[ ! -e "${CONDA_PREFIX}/lib64/libcudart.so" && -f "${CONDA_PREFIX}/targets/x86_64-linux/lib/libcudart.so" ]]; then',
                '  mkdir -p "${CONDA_PREFIX}/lib64"',
                '  ln -sf "${CONDA_PREFIX}/targets/x86_64-linux/lib/libcudart.so" "${CONDA_PREFIX}/lib64/libcudart.so"',
                "fi",
                "if command -v module >/dev/null 2>&1; then module load gcc/11 2>/dev/null || module load gcc/9 2>/dev/null || true; fi",
                'if [[ -n "${_NVIDIA_LD}" ]]; then export LD_LIBRARY_PATH="${_NVIDIA_LD}:${LD_LIBRARY_PATH}"; fi',
                "unset PYTORCH_CUDA_ALLOC_CONF",
                f'export SGLANG_MEM_FRACTION_STATIC="{mem_fraction}"',
                "export SGLANG_ATTENTION_BACKEND=triton",
                "export SGLANG_SAMPLING_BACKEND=pytorch",
                f'export SGLANG_REASONING_PARSER="{reasoning_parser or "deepseek-r1"}"',
                f'export SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH="{"1" if disable_cuda_graph else "0"}"',
                'python -c "import mamba_ssm, causal_conv1d" >/dev/null',
            ]
        )
    elif conda_env == "sglang":
        lines.extend(
            [
                '_NVIDIA_LIB_ROOT="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia"',
                '_NVIDIA_LD=""',
                'if [[ -d "${_NVIDIA_LIB_ROOT}" ]]; then',
                '  for _lib in "${_NVIDIA_LIB_ROOT}"/*/lib; do [[ -d "${_lib}" ]] && _NVIDIA_LD="${_NVIDIA_LD:+${_NVIDIA_LD}:}${_lib}"; done',
                "fi",
                'export LD_LIBRARY_PATH="${_NVIDIA_LD:+${_NVIDIA_LD}:}${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"',
                "if [[ -d /usr/local/cuda-12.6 ]]; then",
                "  export CUDA_HOME=/usr/local/cuda-12.6",
                '  export PATH="${CUDA_HOME}/bin:${PATH}"',
                '  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"',
                "elif [[ -d /usr/local/cuda-12.8 ]]; then",
                "  export CUDA_HOME=/usr/local/cuda-12.8",
                '  export PATH="${CUDA_HOME}/bin:${PATH}"',
                '  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"',
                "fi",
                'if [[ -n "${_NVIDIA_LD}" ]]; then export LD_LIBRARY_PATH="${_NVIDIA_LD}:${LD_LIBRARY_PATH}"; fi',
                "if command -v module >/dev/null 2>&1; then module load gcc/11 2>/dev/null || module load gcc/9 2>/dev/null || true; fi",
                f'export SGLANG_MEM_FRACTION_STATIC="{mem_fraction}"',
                "export SGLANG_ATTENTION_BACKEND=triton",
                "export SGLANG_SAMPLING_BACKEND=pytorch",
            ]
        )
    else:
        lines.append('export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"')
    return "\n".join(lines)


def render(model_key: str, combo: str, short: str) -> str:
    m = get_model_config(model_key)
    ov = model_launch_overrides(model_key)
    backend = ov.get("backend", m.backend)
    mem_fraction = ov.get("mem_fraction_static", 0.80)
    reasoning_parser = m.reasoning_parser or ""
    disable_cuda_graph = bool(ov.get("disable_piecewise_cuda_graph"))
    tier = TIER[model_key]
    score_batch = SCORE_BATCH[tier]
    gen_batch = GEN_BATCH[tier]
    exclude = "#SBATCH --exclude=gpua800n13,gpua800n21\n" if model_key in SBATCH_EXCLUDE else ""
    job_name = f"da22omrans_{combo}_{short}".replace(".", "")[:32]
    conda = _conda_block(model_key, m.conda_env, backend, mem_fraction, reasoning_parser, disable_cuda_graph)

    extra = ""
    if backend == "sglang":
        extra += f"\nEXTRA_ARGS+=(--attention-backend triton --sampling-backend pytorch --mem-fraction-static {mem_fraction})"
        if disable_cuda_graph:
            extra += "\nEXTRA_ARGS+=(--disable-piecewise-cuda-graph)"
        if reasoning_parser:
            extra += f'\nEXTRA_ARGS+=(--reasoning-parser "{reasoning_parser}")'

    return f"""#!/bin/bash
{exclude}#SBATCH --job-name={job_name}
#SBATCH --output=log/data_analysis/22_omr_answer/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=3:00:00
set -euo pipefail

# OMR answer-only teacher_prefix ({combo} / {model_key})
# Same 2048 problems as OMR full-solution analysis; privilege=answer (GT).
# Seq ≤2048 (prompt+completion 1024); high batch for short answer teacher.

BASE_DIR=${{BASE_DIR:-${{SLURM_SUBMIT_DIR:-$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../../.." && pwd)}}}}
JOB_TAG=${{SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}
RUN_SUFFIX="omr_answer"
OUTPUT_DIR=${{OUTPUT_DIR:-${{BASE_DIR}}/scripts/data_analysis/outputs/teacher_prefix_omr_answer/{model_key}/${{RUN_SUFFIX}}_${{JOB_TAG}}}}
DATASET_PATH=${{DATASET_PATH:-{SHARED_DS}}}

TASK="teacher_prefix"
MODEL_KEY="{model_key}"
COMBO="{combo}"
MODEL_PATH="{m.model_path}"
CONDA_ENV="{m.conda_env}"
BACKEND="{backend}"
NUM_PROMPTS=${{NUM_PROMPTS:-2048}}
N_ROLLOUTS=${{N_ROLLOUTS:-2}}
MAX_PROMPT=${{MAX_PROMPT:-1024}}
MAX_COMPLETION=${{MAX_COMPLETION:-1024}}
SCORE_BATCH=${{SCORE_BATCH:-{score_batch}}}
GEN_BATCH_HINT=${{GEN_BATCH_HINT:-{gen_batch}}}

mkdir -p "${{OUTPUT_DIR}}" "${{BASE_DIR}}/log/data_analysis/22_omr_answer"

if [[ ! -f "${{DATASET_PATH}}" ]]; then
  echo "[error] missing shared OMR answer parquet: ${{DATASET_PATH}}" >&2
  echo "[error] run: sbatch scripts/data/prepare_omr_answer_shared2048.sh" >&2
  exit 1
fi

{conda}

export PYTHONPATH="${{BASE_DIR}}/src:${{BASE_DIR}}/scripts/data_analysis:${{PYTHONPATH:-}}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=${{HF_HOME:-${{BASE_DIR}}/.cache/huggingface}}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_CONFIGURE_LOGGING=0

EXTRA_ARGS=(
  --task "${{TASK}}"
  --model-key "${{MODEL_KEY}}"
  --combo "${{COMBO}}"
  --model-path "${{MODEL_PATH}}"
  --dataset-path "${{DATASET_PATH}}"
  --output-dir "${{OUTPUT_DIR}}"
  --num-prompts "${{NUM_PROMPTS}}"
  --n-rollouts "${{N_ROLLOUTS}}"
  --max-prompt-length "${{MAX_PROMPT}}"
  --max-completion-length "${{MAX_COMPLETION}}"
  --temperature 1.1
  --top-p 0.95
  --top-k 20
  --score-batch-size "${{SCORE_BATCH}}"
  --gen-batch-hint "${{GEN_BATCH_HINT}}"
  --backend "${{BACKEND}}"
  --gpu-memory-utilization 0.95
  --seed 42
  --teacher-prefixes answer
  --no-save-token-metrics
){extra}

echo "[analysis] task=${{TASK}} model=${{MODEL_KEY}} combo=${{COMBO}} prefix=answer backend=${{BACKEND}}"
echo "[analysis] dataset=${{DATASET_PATH}}"
echo "[analysis] output=${{OUTPUT_DIR}}"
echo "[analysis] score_batch=${{SCORE_BATCH}} gen_batch_hint=${{GEN_BATCH_HINT}}"

if [[ "${{SKIP_GENERATE:-0}}" != "1" ]]; then
  echo "[analysis] ===== phase 1: generate ====="
  python "${{BASE_DIR}}/scripts/data_analysis/run_opsd_analysis.py" "${{EXTRA_ARGS[@]}}" --skip-score
else
  echo "[analysis] ===== phase 1: skipped (SKIP_GENERATE=1; reuse rollouts.jsonl) ====="
fi

echo "[analysis] ===== phase 2: score answer ====="
python "${{BASE_DIR}}/scripts/data_analysis/run_opsd_analysis.py" "${{EXTRA_ARGS[@]}}" --skip-generate

echo "[analysis] done -> ${{OUTPUT_DIR}}"
ls -lah "${{OUTPUT_DIR}}"
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for model_key, combo, short in MODELS:
        name = f"analyze_{combo}_{model_key.replace('.', '_')}.sh"
        path = OUT_DIR / name
        path.write_text(render(model_key, combo, short), encoding="utf-8")
        path.chmod(0o755)
        written.append(str(path))
        print(f"wrote {path}")

    submit = OUT_DIR / "submit_all.sh"
    submit.write_text(
        """#!/bin/bash
# Submit all 10 OMR answer-only teacher_prefix jobs.
# Requires shared parquet from scripts/data/prepare_omr_answer_shared2048.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
DS="${ROOT}/data/openmathreasoning/preprocessed/omr.correct.answer.shared2048.seed42.maxprompt1024.parquet"
if [[ ! -f "${DS}" ]]; then
  echo "[error] missing ${DS}" >&2
  echo "[error] run prepare first: sbatch scripts/data/prepare_omr_answer_shared2048.sh" >&2
  exit 1
fi
mkdir -p log/data_analysis/22_omr_answer
for s in "$(dirname "${BASH_SOURCE[0]}")"/analyze_*.sh; do
  echo "[submit] $(basename "${s}")"
  sbatch --chdir="${ROOT}" "${s}"
done
echo "[done] submitted OMR answer-only analysis jobs"
""",
        encoding="utf-8",
    )
    submit.chmod(0o755)
    print(f"wrote {submit}")
    print(f"total scripts: {len(written)}")


if __name__ == "__main__":
    main()
