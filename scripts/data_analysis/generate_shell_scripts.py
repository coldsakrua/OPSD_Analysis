#!/usr/bin/env python3
"""Generate per-model/combo shell scripts for sections 2.1–2.5."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
from common.model_registry import ENTROPY_BUCKETS, SECTION_MODEL_COMBOS, TEACHER_PREFIXES

SBATCH_HEADER = """#!/bin/bash
{sbatch_extra}#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# {description}
# Rollout length: 1024 (override MAX_COMPLETION).
# Metrics: KL/JSD (beta=0.0), top-k KL (k=1,16), log-ratio, argmax preference, SNR, loss-dominant tokens.

BASE_DIR=${{BASE_DIR:-${{SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}}}
export TASK={task}
export MODEL_KEY={model_key}
export COMBO={combo}
{extra_exports}
source "${{BASE_DIR}}/scripts/data_analysis/_common_launch.sh"
"""

SBATCH_HEADER_25 = """#!/bin/bash
{sbatch_extra}#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
set -euo pipefail

# {description}
# Rollout length: 6144 (match length training; override MAX_COMPLETION).
# Length windows: 0-128, 128-256, 256-512, 512-1024, 1024-2048, 2048-4096, 4096-6144.
# Metrics: KL/JSD (beta=0.0), top-k KL (k=1,16), log-ratio, argmax preference, SNR, loss-dominant tokens.

BASE_DIR=${{BASE_DIR:-${{SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}}}
export TASK={task}
export MODEL_KEY={model_key}
export COMBO={combo}
export MAX_COMPLETION=6144
{extra_exports}
source "${{BASE_DIR}}/scripts/data_analysis/_common_launch.sh"
"""

SBATCH_SGLANG = "#SBATCH --exclude=gpua800n13,gpua800n21\n"


def write_script(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def job_name(section: str, model: str, combo: str, suffix: str = "") -> str:
    base = f"da_{section}_{combo}_{model}"
    if suffix:
        base = f"{base}_{suffix}"
    return base.replace(".", "").replace("_", "")[:32]


def gen_21() -> list[Path]:
    created = []
    for model, combos in SECTION_MODEL_COMBOS["2.1"].items():
        for combo in combos:
            name = f"analyze_{combo}_{model}.sh"
            path = ROOT / "2.1_combinations" / name
            log_dir = f"log/data_analysis/2.1_combinations/{model}"
            sbatch_extra = SBATCH_SGLANG if model in ("olmo3_7b_instruct", "olmo3_7b_think", "qwen3.5_4b") else ""
            content = SBATCH_HEADER.format(
                job_name=job_name("21", model, combo),
                log_dir=log_dir,
                description=f"2.1 student/teacher combo {combo} on {model}",
                task="combinations",
                model_key=model,
                combo=combo,
                extra_exports="",
                sbatch_extra=sbatch_extra,
            )
            write_script(path, content)
            created.append(path)
    return created


def gen_22() -> list[Path]:
    created = []
    for model, combos in SECTION_MODEL_COMBOS["2.2"].items():
        for combo in combos:
            name = f"analyze_{combo}_{model}.sh"
            path = ROOT / "2.2_teacher_prefix" / name
            log_dir = f"log/data_analysis/2.2_teacher_prefix/{model}"
            sbatch_extra = SBATCH_SGLANG if model.startswith("olmo") else ""
            content = SBATCH_HEADER.format(
                job_name=job_name("22", model, combo),
                log_dir=log_dir,
                description=f"2.2 teacher prefix variants (sol/answer/irr) {combo} {model}",
                task="teacher_prefix",
                model_key=model,
                combo=combo,
                extra_exports="",
                sbatch_extra=sbatch_extra,
            )
            write_script(path, content)
            created.append(path)
    return created


def gen_23() -> list[Path]:
    created = []
    for model, combos in SECTION_MODEL_COMBOS["2.3"].items():
        combo = combos[0]
        for bucket in ENTROPY_BUCKETS:
            name = f"analyze_{combo}_{bucket}_{model}.sh"
            path = ROOT / "2.3_entropy" / name
            log_dir = f"log/data_analysis/2.3_entropy/{model}"
            sbatch_extra = SBATCH_SGLANG if model.startswith("olmo") else ""
            content = SBATCH_HEADER.format(
                job_name=job_name("23", model, combo, bucket),
                log_dir=log_dir,
                description=f"2.3 entropy bucket {bucket} {combo} {model}",
                task="entropy",
                model_key=model,
                combo=combo,
                extra_exports=f'export ENTROPY_BUCKET="{bucket}"',
                sbatch_extra=sbatch_extra,
            )
            write_script(path, content)
            created.append(path)
    return created


def gen_24() -> list[Path]:
    created = []
    for model, combos in SECTION_MODEL_COMBOS["2.4"].items():
        combo = combos[0]
        name = f"analyze_{combo}_{model}.sh"
        path = ROOT / "2.4_other_models" / name
        log_dir = f"log/data_analysis/2.4_other_models/{model}"
        sbatch_extra = SBATCH_SGLANG if model in ("falcon_h1r_7b", "mimo_7b_rl") else ""
        content = SBATCH_HEADER.format(
            job_name=job_name("24", model, combo),
            log_dir=log_dir,
            description=f"2.4 additional model {model} {combo}",
            task="combinations",
            model_key=model,
            combo=combo,
            extra_exports="",
            sbatch_extra=sbatch_extra,
        )
        write_script(path, content)
        created.append(path)
    return created


def gen_25() -> list[Path]:
    created = []
    for model, combos in SECTION_MODEL_COMBOS["2.5"].items():
        combo = combos[0]
        name = f"analyze_{combo}_{model}.sh"
        path = ROOT / "2.5_length_windows" / name
        log_dir = f"log/data_analysis/2.5_length_windows/{model}"
        sbatch_extra = SBATCH_SGLANG if model.startswith("olmo") else ""
        content = SBATCH_HEADER_25.format(
            job_name=job_name("25", model, combo),
            log_dir=log_dir,
            description=f"2.5 length-window analysis {combo} {model}",
            task="length_windows",
            model_key=model,
            combo=combo,
            extra_exports="",
            sbatch_extra=sbatch_extra,
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
    print(f"generated {len(all_paths)} scripts")


if __name__ == "__main__":
    main()
