"""Answer accuracy for on-policy rollouts (boxed + math_verify, eval-aligned)."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Optional

try:
    from math_verify import parse, verify

    _HAS_MATH_VERIFY = True
except Exception:
    _HAS_MATH_VERIFY = False

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_THINK_OPENERS = ("<think>",)
_RELAXED_MATH_ANSWER_PATTERNS = (
    r"[Tt]he answer is:?\s*\$([^$]+)\$",
    r"[Tt]he answer is:?\s*\\boxed\{([^}]+)\}",
    r"[Tt]he answer is:?\s*(\\frac\{[^}]+\}\{[^}]+\})",
    r"[Tt]he answer is:?\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)",
    r"[Ff]inal answer:?\s*\$([^$]+)\$",
    r"[Ff]inal answer:?\s*\\boxed\{([^}]+)\}",
    r"答案是:?\s*\$([^$]+)\$",
    r"答案是:?\s*\\boxed\{([^}]+)\}",
    r"答案是:?\s*(\d+)",
)


def extract_boxed_answer(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    num_left_braces = 0
    right_brace_idx = None
    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[7:-1].strip()
    return None


def _strip_thinking_for_eval(text: str) -> str:
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    for opener in _THINK_OPENERS:
        idx = cleaned.lower().find(opener.lower())
        if idx >= 0:
            cleaned = cleaned[:idx]
    return cleaned.strip()


def extract_relaxed_math_answer(text: str) -> Optional[str]:
    if not text:
        return None
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    best: Optional[str] = None
    best_pos = -1
    for pat in _RELAXED_MATH_ANSWER_PATTERNS:
        for m in re.finditer(pat, text):
            if m.start() >= best_pos:
                best_pos = m.start()
                best = m.group(1).strip()
    return best


def extract_math_answer(text: str, *, relaxed: bool = False) -> Optional[str]:
    stripped = _strip_thinking_for_eval(text or "")
    candidates = [stripped, text or ""] if stripped != (text or "") else [text or ""]
    for cand in candidates:
        if not cand:
            continue
        ans = extract_relaxed_math_answer(cand) if relaxed else extract_boxed_answer(cand)
        if ans is not None:
            return ans
    return None


def normalize_math_ground_truth(ground_truth: str) -> str:
    gt = str(ground_truth or "").strip()
    if not gt:
        return gt
    inner = extract_boxed_answer(gt)
    if inner is not None:
        gt = inner
    elif gt.startswith("\\boxed{") and gt.endswith("}"):
        gt = gt[7:-1].strip()
    if re.fullmatch(r"0*\d+", gt):
        gt = gt.lstrip("0") or "0"
    return gt


def grade_answer(predicted: Optional[str], ground_truth: str) -> bool:
    if predicted is None:
        return False
    gt = normalize_math_ground_truth(ground_truth)
    if _HAS_MATH_VERIFY:
        try:
            pred_w = predicted if "$" in predicted else f"${predicted}$"
            gt_w = gt if "$" in gt else f"${gt}$"
            pred_parsed = parse(pred_w, fallback_mode="no_fallback")
            gt_parsed = parse(gt_w, fallback_mode="no_fallback")
            return bool(verify(gt_parsed, pred_parsed, timeout_seconds=5))
        except Exception:
            pass
    pred_norm = predicted.replace("$", "").replace(" ", "").lower().strip()
    gt_norm = gt.replace("$", "").replace(" ", "").lower().strip()
    return pred_norm == gt_norm


def score_rollout_accuracy(
    rollouts: list[dict[str, Any]],
    *,
    relaxed: bool = False,
) -> list[dict[str, Any]]:
    """Per-rollout accuracy records (pred / correct / extract meta)."""
    rows: list[dict[str, Any]] = []
    for r in rollouts:
        gt = str(r.get("answer") or "").strip()
        text = str(r.get("completion_text") or "")
        pred = extract_math_answer(text, relaxed=relaxed)
        correct = bool(grade_answer(pred, gt)) if gt else False
        rows.append(
            {
                "row_id": r.get("row_id"),
                "rollout_idx": r.get("rollout_idx"),
                "answer": gt,
                "pred": pred,
                "correct": correct,
                "has_pred": pred is not None,
                "completion_len": r.get("completion_len"),
                "finish_reason": r.get("finish_reason"),
            }
        )
    return rows


def aggregate_accuracy(
    acc_rows: list[dict[str, Any]],
    *,
    n_rollouts: int,
) -> dict[str, Any]:
    """Aggregate mean@n / pass@k style metrics over prompts."""
    by_prompt: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in acc_rows:
        by_prompt[row["row_id"]].append(row)

    n_prompts = len(by_prompt)
    n_gens = len(acc_rows)
    n_correct = sum(1 for r in acc_rows if r.get("correct"))
    n_has_pred = sum(1 for r in acc_rows if r.get("has_pred"))

    per_prompt: list[dict[str, Any]] = []
    pass_at: dict[int, int] = {1: 0, 2: 0, 4: 0}
    for row_id, rows in by_prompt.items():
        rows_sorted = sorted(rows, key=lambda x: int(x.get("rollout_idx") or 0))
        corrects = [bool(r.get("correct")) for r in rows_sorted]
        mean_correct = float(sum(corrects) / len(corrects)) if corrects else 0.0
        per_prompt.append(
            {
                "row_id": row_id,
                "n": len(corrects),
                "n_correct": int(sum(corrects)),
                "mean_correct": mean_correct,
                "any_correct": any(corrects),
            }
        )
        for k in pass_at:
            if any(corrects[:k]):
                pass_at[k] += 1

    mean_at_n = float(n_correct / n_gens) if n_gens else 0.0
    prompt_mean = (
        float(sum(p["mean_correct"] for p in per_prompt) / n_prompts) if n_prompts else 0.0
    )
    return {
        "n_prompts": n_prompts,
        "n_generations": n_gens,
        "n_rollouts_requested": n_rollouts,
        "n_correct": n_correct,
        "mean_accuracy": mean_at_n,
        "mean_accuracy_pct": 100.0 * mean_at_n,
        "prompt_mean_accuracy": prompt_mean,
        "prompt_mean_accuracy_pct": 100.0 * prompt_mean,
        "format_rate": float(n_has_pred / n_gens) if n_gens else 0.0,
        "format_rate_pct": 100.0 * (n_has_pred / n_gens) if n_gens else 0.0,
        "pass_at": {
            str(k): {
                "n_solved": pass_at[k],
                "rate": float(pass_at[k] / n_prompts) if n_prompts else 0.0,
                "rate_pct": 100.0 * (pass_at[k] / n_prompts) if n_prompts else 0.0,
            }
            for k in sorted(pass_at)
            if k <= max(n_rollouts, 1)
        },
    }
