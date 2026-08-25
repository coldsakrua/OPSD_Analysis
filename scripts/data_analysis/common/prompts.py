"""Prompt sampling via SelfDistillationDataCollator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from data_collator import SelfDistillationDataCollator  # noqa: E402

from .model_registry import COMBO_THINK, privilege_for_prefix


def make_collator(
    tokenizer: Any,
    *,
    combo: str,
    max_prompt_length: int,
    privilege_mode: str = "opsd",
    teacher_privilege_field: str = "solution",
) -> SelfDistillationDataCollator:
    student_thinking, teacher_thinking = COMBO_THINK[combo]
    return SelfDistillationDataCollator(
        tokenizer=tokenizer,
        max_length=max_prompt_length + 1024,
        max_prompt_length=max_prompt_length,
        privilege_mode=privilege_mode,
        teacher_privilege_field=teacher_privilege_field,
        student_thinking=student_thinking,
        teacher_thinking=teacher_thinking,
    )


def load_prompt_samples(
    dataset_path: str | Path,
    tokenizer: Any,
    *,
    combo: str,
    num_prompts: int,
    max_prompt_length: int,
    seed: int,
    teacher_prefix: str = "sol",
) -> list[dict[str, Any]]:
    """Sample problems that fit under student/teacher prompts."""
    privilege_mode, privilege_field = privilege_for_prefix(teacher_prefix)
    collator = make_collator(
        tokenizer,
        combo=combo,
        max_prompt_length=max_prompt_length,
        privilege_mode=privilege_mode,
        teacher_privilege_field=privilege_field,
    )

    df = pd.read_parquet(dataset_path, columns=["problem", "solution", "answer"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))

    out: list[dict[str, Any]] = []
    for idx in tqdm(order, desc="sample prompts", total=len(order)):
        row = df.iloc[int(idx)]
        feature = {
            "problem": str(row["problem"]).strip(),
            "solution": str(row["solution"]).strip() if row["solution"] is not None else "",
            "answer": str(row["answer"]).strip() if row["answer"] is not None else "",
        }
        if not feature["problem"] or not collator.fits(feature):
            continue
        student_prompt, teacher_prompt = collator.format_prompts(feature)
        out.append(
            {
                "row_id": int(idx),
                "problem": feature["problem"],
                "solution": feature["solution"],
                "answer": feature["answer"],
                "student_prompt": student_prompt,
                "teacher_prompt": teacher_prompt,
                "student_prompt_len": len(tokenizer(student_prompt, add_special_tokens=False)["input_ids"]),
                "teacher_prompt_len": len(tokenizer(teacher_prompt, add_special_tokens=False)["input_ids"]),
                "combo": combo,
                "teacher_prefix": teacher_prefix,
            }
        )
        if len(out) >= num_prompts:
            break

    if len(out) < num_prompts:
        raise RuntimeError(f"only {len(out)} prompts fit (need {num_prompts})")
    return out


def load_multi_prefix_samples(
    dataset_paths: dict[str, str | Path],
    tokenizer: Any,
    *,
    combo: str,
    num_prompts: int,
    max_prompt_length: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample prompts shared across teacher prefix variants (2.2)."""
    collators = {
        name: make_collator(
            tokenizer,
            combo=combo,
            max_prompt_length=max_prompt_length,
            privilege_mode=privilege_for_prefix(name)[0],
            teacher_privilege_field=privilege_for_prefix(name)[1],
        )
        for name in dataset_paths
    }

    # Binding dataset: use sol path rows
    bind_path = dataset_paths["sol"]
    df = pd.read_parquet(bind_path, columns=["problem", "solution", "answer"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))

    out: list[dict[str, Any]] = []
    for idx in tqdm(order, desc="sample multi-prefix", total=len(order)):
        row = df.iloc[int(idx)]
        feature = {
            "problem": str(row["problem"]).strip(),
            "solution": str(row["solution"]).strip() if row["solution"] is not None else "",
            "answer": str(row["answer"]).strip() if row["answer"] is not None else "",
        }
        if not feature["problem"]:
            continue
        if not all(c.fits(feature) for c in collators.values()):
            continue

        student_prompt, _ = collators["sol"].format_prompts(feature)
        rec: dict[str, Any] = {
            "row_id": int(idx),
            "problem": feature["problem"],
            "solution": feature["solution"],
            "answer": feature["answer"],
            "student_prompt": student_prompt,
            "student_prompt_len": len(tokenizer(student_prompt, add_special_tokens=False)["input_ids"]),
            "combo": combo,
        }
        for name, col in collators.items():
            _, tp = col.format_prompts(feature)
            rec[f"teacher_prompt_{name}"] = tp
            rec[f"teacher_prompt_len_{name}"] = len(
                tokenizer(tp, add_special_tokens=False)["input_ids"]
            )
        out.append(rec)
        if len(out) >= num_prompts:
            break

    if len(out) < num_prompts:
        raise RuntimeError(f"only {len(out)} multi-prefix prompts fit (need {num_prompts})")
    return out
