"""CAST advantage verification logging (stdout + JSONL)."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _verify_json_path() -> str | None:
    path = os.environ.get("CAST_VERIFY_JSON", "").strip()
    return path or None


def _append_json_record(record: dict[str, Any]) -> None:
    path = _verify_json_path()
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _group_label(
    idx: int,
    *,
    all_correct: torch.Tensor,
    all_wrong: torch.Tensor,
    mixed: torch.Tensor,
) -> str:
    if bool(all_correct[idx, 0].item()):
        return "all_correct"
    if bool(all_wrong[idx, 0].item()):
        return "all_wrong"
    if bool(mixed[idx, 0].item()):
        return "mixed"
    return "unknown"


def _sample_example_indices(
    n_rows: int,
    *,
    all_correct: torch.Tensor,
    all_wrong: torch.Tensor,
    mixed: torch.Tensor,
    max_examples: int,
) -> list[int]:
    picked: list[int] = []
    buckets: list[tuple[str, list[int]]] = [
        ("all_correct", [i for i in range(n_rows) if bool(all_correct[i, 0].item())]),
        ("all_wrong", [i for i in range(n_rows) if bool(all_wrong[i, 0].item())]),
        ("mixed", [i for i in range(n_rows) if bool(mixed[i, 0].item())]),
    ]
    per_bucket = max(1, max_examples // 3)
    for _name, indices in buckets:
        picked.extend(indices[:per_bucket])
    if len(picked) < max_examples:
        for i in range(n_rows):
            if i not in picked:
                picked.append(i)
            if len(picked) >= max_examples:
                break
    return picked[:max_examples]


def _trajectory_abs_adv_mean(row_idx: int, values: torch.Tensor, mask: torch.Tensor) -> float:
    seq_mask = mask[row_idx].float()
    seq_len = float(seq_mask.sum().item())
    if seq_len <= 0:
        return 0.0
    return float((values[row_idx].abs() * seq_mask).sum().item() / seq_len)


def _format_group_ratio(name: str, count: float, total: float) -> str:
    if total <= 0:
        return f"{name}=0(0.0%)"
    pct = 100.0 * count / total
    return f"{name}={count:.0f}({pct:.1f}%)"


def log_cast_verify(
    *,
    step: int,
    estimator: str,
    grouped_counts: dict[str, float],
    teacher_src: str,
    g: torch.Tensor,
    grpo_base: torch.Tensor,
    shaped_pre_clip: torch.Tensor,
    final_adv: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
    all_correct: torch.Tensor,
    all_wrong: torch.Tensor,
    mixed: torch.Tensor,
    lam: float,
    fallback_scale: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit per-step CAST diagnostics to stderr and optional JSONL."""
    max_examples = int(os.environ.get("CAST_VERIFY_MAX_EXAMPLES", "4") or "4")
    with torch.no_grad():
        m = mask.bool()
        g_vals = g[m]
        adv_vals = final_adv[m]
        shaped_vals = shaped_pre_clip[m]
        grpo_vals = grpo_base[m]

        g_mean = float(g_vals.mean().item()) if g_vals.numel() else 0.0
        g_abs_mean = float(g_vals.abs().mean().item()) if g_vals.numel() else 0.0
        adv_mean = float(adv_vals.mean().item()) if adv_vals.numel() else 0.0
        adv_abs_mean = float(adv_vals.abs().mean().item()) if adv_vals.numel() else 0.0
        shaping_abs_mean = float((shaped_vals - grpo_vals).abs().mean().item()) if grpo_vals.numel() else 0.0
        clip_abs_mean = float((adv_vals - shaped_vals).abs().mean().item()) if adv_vals.numel() else 0.0
        weight_mean = float(weight[m].mean().item()) if m.any() else 1.0
        weight_abs_dev = float((weight[m] - 1.0).abs().mean().item()) if m.any() else 0.0

        n_groups = max(
            grouped_counts.get("all_correct", 0.0)
            + grouped_counts.get("all_wrong", 0.0)
            + grouped_counts.get("mixed", 0.0),
            1.0,
        )
        group_line = " ".join(
            [
                _format_group_ratio("all_correct", grouped_counts.get("all_correct", 0.0), n_groups),
                _format_group_ratio("all_wrong", grouped_counts.get("all_wrong", 0.0), n_groups),
                _format_group_ratio("mixed", grouped_counts.get("mixed", 0.0), n_groups),
            ]
        )

        n_rows = int(mask.shape[0])
        sample_indices = _sample_example_indices(
            n_rows,
            all_correct=all_correct,
            all_wrong=all_wrong,
            mixed=mixed,
            max_examples=max_examples,
        )
        examples: list[dict[str, Any]] = []
        for idx in sample_indices:
            grpo_abs = _trajectory_abs_adv_mean(idx, grpo_base, mask)
            shaped_abs = _trajectory_abs_adv_mean(idx, shaped_pre_clip, mask)
            examples.append(
                {
                    "sample_idx": idx,
                    "group": _group_label(idx, all_correct=all_correct, all_wrong=all_wrong, mixed=mixed),
                    "grpo_abs_adv_mean": grpo_abs,
                    "shaped_abs_adv_mean": shaped_abs,
                }
            )

    summary = (
        f"[CAST verify step={step}] estimator={estimator} "
        f"groups: {group_line} | "
        f"teacher_src={teacher_src} "
        f"g_mean={g_mean:.6f} g_abs_mean={g_abs_mean:.6f} "
        f"adv_mean={adv_mean:.6f} adv_abs_mean={adv_abs_mean:.6f} | "
        f"shaping_abs_mean={shaping_abs_mean:.6f} clip_abs_mean={clip_abs_mean:.6f} | "
        f"teacher_weight_mean={weight_mean:.6f} weight_dev_from_1={weight_abs_dev:.6f} | "
        f"lambda={lam:.4f} fallback_scale={fallback_scale:.4f}"
    )
    logger.info(summary)
    print(summary, file=sys.stderr, flush=True)

    for ex in examples:
        ex_line = (
            f"[CAST verify ex step={step}] idx={ex['sample_idx']} group={ex['group']} "
            f"grpo_abs_adv_mean={ex['grpo_abs_adv_mean']:.4f} "
            f"shaped_abs_adv_mean={ex['shaped_abs_adv_mean']:.4f}"
        )
        print(ex_line, file=sys.stderr, flush=True)

    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "estimator": estimator,
        "teacher_src": teacher_src,
        "groups": grouped_counts,
        "g_mean": g_mean,
        "g_abs_mean": g_abs_mean,
        "adv_mean": adv_mean,
        "adv_abs_mean": adv_abs_mean,
        "shaping_abs_mean": shaping_abs_mean,
        "clip_abs_mean": clip_abs_mean,
        "teacher_weight_mean": weight_mean,
        "teacher_weight_abs_dev_from_1": weight_abs_dev,
        "lambda": lam,
        "fallback_scale": fallback_scale,
        "examples": examples,
    }
    if extra:
        record.update(extra)
    _append_json_record(record)
