#!/usr/bin/env python3
"""Generate self-contained per-model SLURM scripts for sections 2.1–2.6."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
from common.model_registry import (  # noqa: E402
    SECTION_MODEL_COMBOS,
    COTLEN_BANDS,
    COTLEN_MAX_PROMPT,
    cotlen_max_completion,
    get_model_config,
    model_launch_overrides,
    task_default_gen_batch_hint,
    task_default_max_completion,
    task_default_score_batch,
)

MODEL_SHORT = {
    "qwen3_1.7b": "1p7b",
    "qwen3_4b": "4b",
    "qwen3_4b_instruct": "4bi",
    "qwen3_4b_thinking": "4bt",
    "qwen3.5_4b": "q35",
    "qwen3_06b": "0p6b",
    "qwen3_8b": "8b",
    "olmo3_7b_instruct": "olmo7bi",
    "olmo3_7b_think": "olmo7bt",
    "deepseek_r1_1.5b": "ds1p5b",
    "falcon_h1r_7b": "falcon7b",
    "mimo_7b_rl": "mimo7b",
}

SECTION_TAG = {
    "2.1": "21",
    "2.2": "22",
    "2.3": "23",
    "2.4": "24",
    "2.5": "25",
    "2.6": "26",
}

SECTION_DIR = {
    "2.1": "2.1_combinations",
    "2.2": "2.2_teacher_prefix",
    "2.3": "2.3_entropy",
    "2.4": "2.4_other_models",
    "2.5": "2.5_length_windows",
    "2.6": "2.6_cotlen",
}

SBATCH_EXCLUDE = {
    "olmo3_7b_instruct": "#SBATCH --exclude=gpua800n13,gpua800n21\n",
    "olmo3_7b_think": "#SBATCH --exclude=gpua800n13,gpua800n21\n",
    "qwen3.5_4b": "#SBATCH --exclude=gpua800n13,gpua800n21\n",
    "falcon_h1r_7b": "#SBATCH --exclude=gpua800n13,gpua800n21\n",
    "mimo_7b_rl": "#SBATCH --exclude=gpua800n13,gpua800n21\n",
}


def _job_name(section: str, combo: str, model: str, suffix: str = "") -> str:
    tag = SECTION_TAG[section]
    short = MODEL_SHORT.get(model, model.replace("_", "")[:8])
    name = f"da{tag}_{combo}_{short}"
    if suffix:
        name = f"{name}_{suffix}"
    return name.replace(".", "")[:32]


def _conda_setup(conda_env: str, backend: str, model_key: str, mem_fraction: float, reasoning_parser: str, disable_cuda_graph: bool) -> str:
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
    elif conda_env == "qwen3_5":
        lines.extend(
            [
                'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"',
                "if [[ -d /usr/local/cuda-12.8 ]]; then",
                "  export CUDA_HOME=/usr/local/cuda-12.8",
                '  export PATH="${CUDA_HOME}/bin:${PATH}"',
                '  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"',
                "elif command -v module >/dev/null 2>&1; then",
                "  module load cuda/12.8 2>/dev/null || true",
                "fi",
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
            ]
        )
    else:
        lines.append('export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"')

    if backend == "sglang" and conda_env not in ("falcon",):
        lines.extend(
            [
                f'export SGLANG_MEM_FRACTION_STATIC="{mem_fraction}"',
                "export SGLANG_ATTENTION_BACKEND=triton",
                "export SGLANG_SAMPLING_BACKEND=pytorch",
            ]
        )
    return "\n".join(lines)


def render_script(
    *,
    section: str,
    task: str,
    model_key: str,
    combo: str,
    description: str,
    entropy_bucket: str = "",
    cotlen_band: str = "",
    max_prompt: int | None = None,
    max_completion: int | None = None,
    num_prompts: int | None = None,
    n_rollouts: int | None = None,
    time_limit: str = "48:00:00",
    preference_separate: bool = False,
) -> str:
    m = get_model_config(model_key)
    ov = model_launch_overrides(model_key)
    backend = ov.get("backend", m.backend)
    mem_fraction = ov.get("mem_fraction_static", 0.80)
    reasoning_parser = m.reasoning_parser or ""
    disable_cuda_graph = bool(ov.get("disable_piecewise_cuda_graph"))
    max_comp = (
        max_completion
        if max_completion is not None
        else task_default_max_completion(task, model_key if task == "cotlen" else None)
    )
    max_prm = max_prompt if max_prompt is not None else 1024
    n_prompts = num_prompts if num_prompts is not None else 2048
    n_rolls = n_rollouts if n_rollouts is not None else (4 if task == "cotlen" else 2)
    score_batch = task_default_score_batch(task, model_key)
    gen_batch_hint = task_default_gen_batch_hint(task, model_key)

    suffix_parts = [p for p in (entropy_bucket, cotlen_band) if p]
    suffix = "_".join(suffix_parts)
    job_name = _job_name(section, combo, model_key, suffix)
    log_section = section.replace(".", "")
    exclude = SBATCH_EXCLUDE.get(model_key, "")

    run_suffix = combo
    if cotlen_band:
        run_suffix = f"{combo}_{cotlen_band}"
    elif entropy_bucket:
        run_suffix = f"{combo}_{entropy_bucket}"

    extra_args = ""
    if entropy_bucket:
        extra_args += f'\nEXTRA_ARGS+=(--entropy-bucket "{entropy_bucket}")'
    if cotlen_band:
        extra_args += f'\nEXTRA_ARGS+=(--cotlen-band "{cotlen_band}")'
    if backend == "sglang":
        extra_args += f"\nEXTRA_ARGS+=(--attention-backend triton --sampling-backend pytorch --mem-fraction-static {mem_fraction})"
        if disable_cuda_graph:
            extra_args += "\nEXTRA_ARGS+=(--disable-piecewise-cuda-graph)"
        if reasoning_parser:
            extra_args += f'\nEXTRA_ARGS+=(--reasoning-parser "{reasoning_parser}")'

    length_note = ""
    if section == "2.2":
        length_note = (
            "# Prefixes default=long: sol/answer/irrelevant/sol_long (≤12288). "
            "POOL=short → legacy 3 prefixes only.\n"
            "# Short SCORE_BATCH 8/4/2; sol_long auto-scales 8/4/2/1 by params.\n"
        )
        # long pool skips token_metrics; short pool matches legacy (save token metrics).
        extra_args += """
if [[ "${POOL}" == "short" ]]; then
  EXTRA_ARGS+=(--teacher-prefixes sol answer irrelevant_other_sol --save-token-metrics)
else
  EXTRA_ARGS+=(--no-save-token-metrics)
fi"""
    elif section == "2.3":
        length_note = "# Entropy buckets: he20, le20, he80, le80 (single score pass).\n"
    elif section == "2.5":
        length_note = "# Length windows: 0-128 … 4096-6144.\n"
    elif section == "2.6":
        length_note = (
            f"# OT cotlen band={cotlen_band}: prompt=2048 completion={max_comp}; "
            f"n={n_rolls} rollouts.\n"
            "# Default: generate + boxed accuracy only. "
            "Preference/data-analysis: RUN_PREFERENCE=1 (or submit_preference.sh).\n"
        )

    if preference_separate:
        phase_block = """if [[ "${SKIP_GENERATE:-0}" != "1" ]]; then
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
fi"""
    else:
        phase_block = """if [[ "${SKIP_GENERATE:-0}" != "1" ]]; then
  echo "[analysis] ===== phase 1: generate ====="
  python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-score
else
  echo "[analysis] ===== phase 1: skipped (SKIP_GENERATE=1; reuse rollouts.jsonl) ====="
fi

echo "[analysis] ===== phase 2: score ====="
python "${BASE_DIR}/scripts/data_analysis/run_opsd_analysis.py" "${EXTRA_ARGS[@]}" --skip-generate"""

    return f"""#!/bin/bash
{exclude}#SBATCH --job-name={job_name}
#SBATCH --output=log/data_analysis/{log_section}/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time={time_limit}
set -euo pipefail

# {description}
{length_note}# Rollout: temp=1.1 top_p=0.95 top_k=20 max_prompt={max_prm} max_completion={max_comp}

BASE_DIR=${{BASE_DIR:-${{SLURM_SUBMIT_DIR:-$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../../.." && pwd)}}}}
JOB_TAG=${{SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}
# 2.2: POOL=long (default, +sol_long@12288) | POOL=short (legacy sol/answer/irrelevant)
POOL=${{POOL:-long}}
RUN_SUFFIX="{run_suffix}"
if [[ "{section}" == "2.2" && "${{POOL}}" == "short" ]]; then
  RUN_SUFFIX="{run_suffix}_short"
fi
OUTPUT_DIR=${{OUTPUT_DIR:-${{BASE_DIR}}/scripts/data_analysis/outputs/{task}/{model_key}/${{RUN_SUFFIX}}_${{JOB_TAG}}}}

TASK="{task}"
MODEL_KEY="{model_key}"
COMBO="{combo}"
MODEL_PATH="{m.model_path}"
CONDA_ENV="{m.conda_env}"
BACKEND="{backend}"
NUM_PROMPTS=${{NUM_PROMPTS:-{n_prompts}}}
N_ROLLOUTS=${{N_ROLLOUTS:-{n_rolls}}}
MAX_PROMPT=${{MAX_PROMPT:-{max_prm}}}
MAX_COMPLETION=${{MAX_COMPLETION:-{max_comp}}}
SCORE_BATCH=${{SCORE_BATCH:-{score_batch}}}
GEN_BATCH_HINT=${{GEN_BATCH_HINT:-{gen_batch_hint}}}

mkdir -p "${{OUTPUT_DIR}}" "${{BASE_DIR}}/log/data_analysis/{log_section}"

{_conda_setup(m.conda_env, backend, model_key, mem_fraction, reasoning_parser, disable_cuda_graph)}

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
  --gpu-memory-utilization 0.90
  --seed 42
){extra_args}

echo "[analysis] task=${{TASK}} model=${{MODEL_KEY}} combo=${{COMBO}} backend=${{BACKEND}}"
echo "[analysis] output=${{OUTPUT_DIR}}"

{phase_block}

echo "[analysis] done -> ${{OUTPUT_DIR}}"
ls -lah "${{OUTPUT_DIR}}"
"""


def write_script(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def gen_21() -> list[Path]:
    created = []
    for model, combos in SECTION_MODEL_COMBOS["2.1"].items():
        for combo in combos:
            path = ROOT / SECTION_DIR["2.1"] / f"analyze_{combo}_{model}.sh"
            content = render_script(
                section="2.1",
                task="combinations",
                model_key=model,
                combo=combo,
                description=f"2.1 combo {combo} on {model}",
            )
            write_script(path, content)
            created.append(path)
    return created


def gen_22() -> list[Path]:
    created = []
    for model, combos in SECTION_MODEL_COMBOS["2.2"].items():
        for combo in combos:
            path = ROOT / SECTION_DIR["2.2"] / f"analyze_{combo}_{model}.sh"
            content = render_script(
                section="2.2",
                task="teacher_prefix",
                model_key=model,
                combo=combo,
                description=f"2.2 teacher prefix {combo} on {model} (sol/answer/irrelevant/sol_long@12288)",
            )
            write_script(path, content)
            created.append(path)
    return created


def gen_23() -> list[Path]:
    created = []
    for model, combos in SECTION_MODEL_COMBOS["2.3"].items():
        combo = combos[0]
        path = ROOT / SECTION_DIR["2.3"] / f"analyze_{combo}_{model}.sh"
        content = render_script(
            section="2.3",
            task="entropy",
            model_key=model,
            combo=combo,
            description=f"2.3 entropy (he20/le20/he80/le80) {combo} on {model}",
        )
        write_script(path, content)
        created.append(path)
    return created


def gen_24() -> list[Path]:
    created = []
    for model, combos in SECTION_MODEL_COMBOS["2.4"].items():
        combo = combos[0]
        path = ROOT / SECTION_DIR["2.4"] / f"analyze_{combo}_{model}.sh"
        content = render_script(
            section="2.4",
            task="combinations",
            model_key=model,
            combo=combo,
            description=f"2.4 {model} {combo}",
        )
        write_script(path, content)
        created.append(path)
    return created


def gen_25() -> list[Path]:
    created = []
    for model, combos in SECTION_MODEL_COMBOS["2.5"].items():
        combo = combos[0]
        path = ROOT / SECTION_DIR["2.5"] / f"analyze_{combo}_{model}.sh"
        content = render_script(
            section="2.5",
            task="length_windows",
            model_key=model,
            combo=combo,
            description=f"2.5 length windows {combo} on {model}",
            max_completion=6144,
        )
        write_script(path, content)
        created.append(path)
    return created


def gen_26() -> list[Path]:
    created = []
    for model, combos in SECTION_MODEL_COMBOS["2.6"].items():
        combo = combos[0]
        max_comp = cotlen_max_completion(model)
        for band in COTLEN_BANDS:
            path = ROOT / SECTION_DIR["2.6"] / f"analyze_{combo}_{band}_{model}.sh"
            content = render_script(
                section="2.6",
                task="cotlen",
                model_key=model,
                combo=combo,
                cotlen_band=band,
                description=f"2.6 cotlen-{band} {combo} on {model}",
                # prompt=2048 matches train; completion matches project eval max_new_tokens.
                max_prompt=COTLEN_MAX_PROMPT,
                max_completion=max_comp,
                num_prompts=2048,
                n_rollouts=4,
                # SGLang + eval-length completions; think easy/hard can exceed 48h.
                time_limit="7-00:00:00",
                preference_separate=True,
            )
            write_script(path, content)
            created.append(path)
    return created


def main() -> None:
    all_paths: list[Path] = []
    all_paths.extend(gen_21())
    all_paths.extend(gen_22())
    all_paths.extend(gen_23())
    all_paths.extend(gen_24())
    all_paths.extend(gen_25())
    all_paths.extend(gen_26())
    print(f"generated {len(all_paths)} self-contained scripts")


if __name__ == "__main__":
    main()
