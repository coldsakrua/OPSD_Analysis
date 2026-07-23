"""Shared DAPO → OPSD dataset loading / field normalization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk


def content_to_text(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(x for x in parts if x).strip()
    return "" if content is None else str(content).strip()


def extract_problem(prompt: Any) -> str:
    if isinstance(prompt, list):
        users = [x for x in prompt if isinstance(x, dict) and str(x.get("role", "user")).lower() == "user"]
        text = content_to_text((users[-1] if users else prompt[-1]).get("content", "")) if prompt else ""
    elif isinstance(prompt, dict):
        text = content_to_text(prompt.get("content", prompt))
    else:
        text = str(prompt or "").strip()
    text = re.sub(
        r"^\s*Solve\s+the\s+following(?:\s+math)?\s+problem\s*,?\s+step\s+by\s+step\s*[.:]?\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"\s*Please\s+reason\s+step\s+by\s+step.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return text.strip()


def extract_solution(row: dict[str, Any]) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"]).strip()
    for key in ("ground_truth", "answer", "solution", "target"):
        if row.get(key) is not None and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def is_opsd_ready(dataset: Dataset) -> bool:
    cols = set(dataset.column_names)
    return "problem" in cols and "solution" in cols


def load_training_dataset(path: str) -> Dataset:
    source = Path(path)
    if source.is_dir() and (source / "dataset_info.json").exists():
        dataset = load_from_disk(str(source))
        return dataset["train"] if hasattr(dataset, "keys") and "train" in dataset else dataset
    files = [x.strip() for x in path.split(",") if x.strip()]
    return load_dataset("parquet", data_files=files, split="train")


def dataset_meta_path(path: str) -> Path:
    return Path(path).with_suffix(Path(path).suffix + ".meta.json")


def prompt_length_filter_applied(
    path: str,
    *,
    privilege_mode: str,
    enable_thinking: bool,
    max_prompt_length: int,
    model_path: str,
) -> bool:
    """True when sibling .meta.json records a matching offline prompt-length filter."""
    meta_file = dataset_meta_path(path)
    if not meta_file.is_file():
        return False
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not meta.get("prompt_length_filtered"):
        return False
    return (
        str(meta.get("privilege_mode")) == str(privilege_mode)
        and bool(meta.get("enable_thinking")) == bool(enable_thinking)
        and int(meta.get("max_prompt_length", -1)) == int(max_prompt_length)
        and str(meta.get("model_path", "")) == str(model_path)
    )


def normalize_dataset(dataset: Dataset, *, num_proc: int | None = None) -> Dataset:
    """Convert DAPO-style rows to {problem, solution}. No-op if already ready."""
    if is_opsd_ready(dataset):
        print("[dataset] already has problem/solution; skip field normalization", flush=True)
        return dataset

    def convert(row: dict[str, Any]) -> dict[str, str]:
        prompt = row.get("prompt", row.get("problem", row.get("question", "")))
        return {"problem": extract_problem(prompt), "solution": extract_solution(row)}

    map_kwargs: dict[str, Any] = {"desc": "Normalizing DAPO fields", "remove_columns": dataset.column_names}
    if num_proc and num_proc > 1:
        map_kwargs["num_proc"] = num_proc
    dataset = dataset.map(convert, **map_kwargs)
    return dataset.filter(
        lambda row: bool(str(row["problem"]).strip()) and bool(str(row["solution"]).strip()),
        desc="Dropping empty problems/answers",
        **({"num_proc": num_proc} if num_proc and num_proc > 1 else {}),
    )
