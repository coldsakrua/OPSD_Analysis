#!/usr/bin/env python3
"""Generate SLURM scripts for OMR full-solution teacher-prefix analysis (10 models)."""

from __future__ import annotations

import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR.parent))

from common.model_registry import get_model_config, model_launch_overrides  # noqa: E402

# Teacher prompt ≤2k (trimmed full solution); seq ≤3072.
# Full [B,S,V] fp32 logits dominate VRAM; keep score batch conservative.
#   small ≤1.7B (~3.5GB w) → B=8; medium 4B (~8GB) → B=4; large 7B (~14GB) → B=2
SCORE_BATCH = {"small": 8, "medium": 4, "large": 2}
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
    "omr.opsd.solution.postthink.shared2048.seed42.maxprompt2048.parquet"
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
    job_name = f"da22omr_{combo}_{short}".replace(".", "")[:32]
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
#SBATCH --output=log/data_analysis/22_omr/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=3:00:00
set -euo pipefail

# OMR full-solution teacher_prefix ({combo} / {model_key})
# Shared 2048 problems × 2 rollouts; privilege=solution (opsd).
# Teacher prompt ≤2048, completion ≤1024 → seq ≤3072.
# SCORE_BATCH/GEN_BATCH calibrated for 2k teacher prompts on A800 80GB.

BASE_DIR=${{BASE_DIR:-${{SLURM_SUBMIT_DIR:-$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../../.." && pwd)}}}}
JOB_TAG=${{SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}
RUN_SUFFIX="omr_solution"
OUTPUT_DIR=${{OUTPUT_DIR:-${{BASE_DIR}}/scripts/data_analysis/outputs/teacher_prefix_omr/{model_key}/${{RUN_SUFFIX}}_${{JOB_TAG}}}}
DATASET_PATH=${{DATASET_PATH:-{SHARED_DS}}}

TASK="teacher_prefix"
MODEL_KEY="{model_key}"
COMBO="{combo}"
MODEL_PATH="{m.model_path}"
CONDA_ENV="{m.conda_env}"
BACKEND="{backend}"
NUM_PROMPTS=${{NUM_PROMPTS:-2048}}
N_ROLLOUTS=${{N_ROLLOUTS:-2}}
MAX_PROMPT=${{MAX_PROMPT:-2048}}
MAX_COMPLETION=${{MAX_COMPLETION:-1024}}
SCORE_BATCH=${{SCORE_BATCH:-{score_batch}}}
GEN_BATCH_HINT=${{GEN_BATCH_HINT:-{gen_batch}}}

mkdir -p "${{OUTPUT_DIR}}" "${{BASE_DIR}}/log/data_analysis/22_omr"

if [[ ! -f "${{DATASET_PATH}}" ]]; then
  echo "[error] missing shared OMR parquet: ${{DATASET_PATH}}" >&2
  echo "[error] run: sbatch scripts/data/prepare_omr_solution_shared2048.sh" >&2
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
  --teacher-prefixes sol
  --no-save-token-metrics
){extra}

echo "[analysis] task=${{TASK}} model=${{MODEL_KEY}} combo=${{COMBO}} prefix=sol backend=${{BACKEND}}"
echo "[analysis] dataset=${{DATASET_PATH}}"
echo "[analysis] output=${{OUTPUT_DIR}}"
echo "[analysis] score_batch=${{SCORE_BATCH}} gen_batch_hint=${{GEN_BATCH_HINT}}"

if [[ "${{SKIP_GENERATE:-0}}" != "1" ]]; then
  echo "[analysis] ===== phase 1: generate ====="
  python "${{BASE_DIR}}/scripts/data_analysis/run_opsd_analysis.py" "${{EXTRA_ARGS[@]}}" --skip-score
else
  echo "[analysis] ===== phase 1: skipped (SKIP_GENERATE=1; reuse rollouts.jsonl) ====="
fi

echo "[analysis] ===== phase 2: score sol ====="
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
# Submit all 10 OMR full-solution teacher_prefix jobs.
# Requires shared parquet from scripts/data/prepare_omr_solution_shared2048.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
DS="${ROOT}/data/openmathreasoning/preprocessed/omr.opsd.solution.postthink.shared2048.seed42.maxprompt4096.parquet"
if [[ ! -f "${DS}" ]]; then
  echo "[error] missing ${DS}" >&2
  echo "[error] run prepare first: sbatch scripts/data/prepare_omr_solution_shared2048.sh" >&2
  exit 1
fi
mkdir -p log/data_analysis/22_omr
for s in "$(dirname "${BASH_SOURCE[0]}")"/analyze_*.sh; do
  echo "[submit] $(basename "${s}")"
  sbatch --chdir="${ROOT}" "${s}"
done
echo "[done] submitted OMR full-solution analysis jobs"
""",
        encoding="utf-8",
    )
    submit.chmod(0o755)
    print(f"wrote {submit}")
    print(f"total scripts: {len(written)}")


if __name__ == "__main__":
    main()
