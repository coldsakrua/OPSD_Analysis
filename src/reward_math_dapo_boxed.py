"""DAPO-Math reward: prefer \\boxed{}, fall back to Minerva ``Answer:``.

verl's default ``math_dapo.compute_score`` uses Minerva extraction only
(``Answer:``). Qwen-style SFT/GRPO rollouts usually emit ``\\boxed{...}`` and
get reward -1 even when correct. This module is wired as
``custom_reward_function`` for GRPO.
"""

from __future__ import annotations

from typing import Any, Optional

from verl.utils.reward_score.math_dapo import (
    is_correct_minerva,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)


def _boxed_pred(solution_str: str) -> Optional[str]:
    boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return None
    try:
        return normalize_final_answer(remove_boxed(boxed))
    except Exception:
        return None


def compute_score(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: str = "",
    extra_info: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Score a completion. Signature matches DAPORewardManager / default_compute_score."""
    _ = data_source, extra_info, kwargs
    solution_str = solution_str or ""
    gt = normalize_final_answer(str(ground_truth))

    pred = _boxed_pred(solution_str)
    if pred is not None:
        correct = pred == gt
        return {
            "score": 1.0 if correct else -1.0,
            "acc": bool(correct),
            "pred": pred,
            "extract": "boxed",
        }

    # Fall back to Minerva ``Answer:`` (official DAPO format line).
    correct, pred = is_correct_minerva(solution_str, str(ground_truth))
    return {
        "score": 1.0 if correct else -1.0,
        "acc": bool(correct),
        "pred": pred,
        "extract": "answer_colon",
    }
