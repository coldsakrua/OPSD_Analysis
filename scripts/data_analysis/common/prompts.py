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

from .model_registry import (
    LONG_TEACHER_MAX_PROMPT,
    combo_think,
    privilege_for_prefix,
    teacher_prefix_max_prompt,
)

DISTRACTOR_COLUMNS = (
    "distractor_problems",
    "distractor_solutions",
    "distractor_indices",
    "distractor_problem",
    "distractor_solution",
    "distractor_index",
)


def _distractor_lookup(dataset_path: str | Path) -> dict[str, dict[str, Any]]:
    """Map problem text → distractor columns for irrelevant_other_sol prompts."""
    cols = ["problem", *DISTRACTOR_COLUMNS]
    df = pd.read_parquet(dataset_path, columns=cols)
    lookup: dict[str, dict[str, Any]] = {}
    for row in df.itertuples(index=False):
        problem = str(getattr(row, "problem", "") or "").strip()
        if not problem or problem in lookup:
            continue
        extras: dict[str, Any] = {}
        for col in DISTRACTOR_COLUMNS:
            val = getattr(row, col, None)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            extras[col] = val.tolist() if hasattr(val, "tolist") else val
        if extras:
            lookup[problem] = extras
    return lookup


def _merge_distractors(feature: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> bool:
    """Attach distractor fields; return False when lookup has no entry."""
    extras = lookup.get(feature["problem"])
    if not extras:
        return False
    feature.update(extras)
    return True


def make_collator(
    tokenizer: Any,
    *,
    model_key: str,
    combo: str,
    max_prompt_length: int,
    privilege_mode: str = "opsd",
    teacher_privilege_field: str = "solution",
) -> SelfDistillationDataCollator:
    student_thinking, teacher_thinking = combo_think(model_key, combo)
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
    model_key: str,
    combo: str,
    num_prompts: int,
    max_prompt_length: int,
    seed: int,
    teacher_prefix: str = "sol",
    allow_shortfall: bool = False,
) -> list[dict[str, Any]]:
    """Sample problems that fit under student/teacher prompts."""
    privilege_mode, privilege_field = privilege_for_prefix(teacher_prefix)
    collator = make_collator(
        tokenizer,
        model_key=model_key,
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
        msg = f"only {len(out)} prompts fit (need {num_prompts})"
        if not allow_shortfall or not out:
            raise RuntimeError(msg)
        print(f"[sample] WARN {msg}; using all that fit", flush=True)
    return out


def _token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _teacher_len(collator: Any, feature: dict[str, Any]) -> int:
    _, teacher_prompt = collator.format_prompts(feature)[:2]
    return _token_len(collator.teacher_tokenizer, teacher_prompt)


def _truncate_solution_for_teacher(
    collator: Any,
    feature: dict[str, Any],
    *,
    max_prompt_length: int,
) -> dict[str, Any] | None:
    """Copy feature with solution truncated so teacher prompt fits ``max_prompt_length``.

    Used for short ``sol`` when the binding pool allows long solutions (``sol_long``).
    Returns None if even an empty solution cannot fit.
    """
    feat = dict(feature)
    sol = str(feat.get("solution") or "")
    if _teacher_len(collator, feat) <= max_prompt_length:
        return feat
    lo, hi = 0, len(sol)
    best: str | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        feat["solution"] = sol[:mid]
        if _teacher_len(collator, feat) <= max_prompt_length:
            best = sol[:mid]
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        return None
    feat["solution"] = best
    return feat


def load_multi_prefix_samples(
    dataset_paths: dict[str, str | Path],
    tokenizer: Any,
    *,
    model_key: str,
    combo: str,
    num_prompts: int,
    max_prompt_length: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample prompts shared across teacher prefix variants (2.2).

    Student generation stays under ``max_prompt_length`` (default 1024).
    Each teacher prefix uses its own cap (``sol_long`` → 12288; others → max_prompt_length).
    When both ``sol`` and ``sol_long`` are present, bind to the long solution pool and
    truncate ``sol``'s solution so the short teacher prompt still fits.
    """
    prefix_max = {
        name: teacher_prefix_max_prompt(name, default_max_prompt=max_prompt_length)
        for name in dataset_paths
    }
    collators = {
        name: make_collator(
            tokenizer,
            model_key=model_key,
            combo=combo,
            max_prompt_length=prefix_max[name],
            privilege_mode=privilege_for_prefix(name)[0],
            teacher_privilege_field=privilege_for_prefix(name)[1],
        )
        for name in dataset_paths
    }

    # Prefer long sol pool when present so sol_long can exceed the short cap.
    if "sol_long" in dataset_paths:
        bind_path = dataset_paths["sol_long"]
        student_col = collators["sol_long"]
    elif "sol" in dataset_paths:
        bind_path = dataset_paths["sol"]
        student_col = collators["sol"]
    else:
        raise KeyError("teacher_prefix sampling needs sol or sol_long in dataset_paths")

    df = pd.read_parquet(bind_path, columns=["problem", "solution", "answer"])
    distractor_lookup: dict[str, dict[str, Any]] | None = None
    if "irrelevant_other_sol" in dataset_paths:
        distractor_lookup = _distractor_lookup(dataset_paths["irrelevant_other_sol"])
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
        if distractor_lookup is not None and not _merge_distractors(feature, distractor_lookup):
            continue

        student_prompt, _ = student_col.format_prompts(feature)
        student_len = _token_len(tokenizer, student_prompt)
        if student_len > max_prompt_length:
            continue

        teacher_prompts: dict[str, str] = {}
        teacher_lens: dict[str, int] = {}
        ok = True
        for name, col in collators.items():
            feat_for_prefix = feature
            if (
                name == "sol"
                and "sol_long" in collators
                and _teacher_len(col, feature) > prefix_max[name]
            ):
                truncated = _truncate_solution_for_teacher(
                    col, feature, max_prompt_length=prefix_max[name]
                )
                if truncated is None:
                    ok = False
                    break
                feat_for_prefix = truncated
            _, tp = col.format_prompts(feat_for_prefix)
            t_len = _token_len(col.teacher_tokenizer, tp)
            if t_len > prefix_max[name]:
                ok = False
                break
            teacher_prompts[name] = tp
            teacher_lens[name] = t_len
        if not ok:
            continue

        rec: dict[str, Any] = {
            "row_id": int(idx),
            "problem": feature["problem"],
            "solution": feature["solution"],
            "answer": feature["answer"],
            "student_prompt": student_prompt,
            "student_prompt_len": student_len,
            "combo": combo,
            "long_teacher_max_prompt": LONG_TEACHER_MAX_PROMPT,
        }
        for name in collators:
            rec[f"teacher_prompt_{name}"] = teacher_prompts[name]
            rec[f"teacher_prompt_len_{name}"] = teacher_lens[name]
        out.append(rec)
        if len(out) >= num_prompts:
            break

    if len(out) < num_prompts:
        raise RuntimeError(f"only {len(out)} multi-prefix prompts fit (need {num_prompts})")
    return out
