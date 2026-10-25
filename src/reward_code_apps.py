"""APPS / code GRPO reward via ``eval_code`` (extract + sandboxed tests).

Supports both APIs:
  - DAPO / naive: ``compute_score(data_source=, solution_str=, ground_truth=, ...)``
  - BatchRewardManager: ``compute_score(data_sources=, solution_strs=, ground_truths=, ...)``
    → ThreadPool of ``check_correctness`` (each still sandboxed in its own Process).

``ground_truth`` is LCB-normalized APPS ``input_output`` JSON.
Score: ``+1.0`` if all tests pass else ``-1.0``.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

_REPO = Path(__file__).resolve().parents[1]
_EVAL_CODE = _REPO / "eval_code"
if str(_EVAL_CODE) not in sys.path:
    sys.path.insert(0, str(_EVAL_CODE))

from code_extract import extract_lcb_candidate  # noqa: E402
from lcb_runner.evaluation.compute_code_generation_metrics import (  # noqa: E402
    check_correctness,
)

# Leave headroom for Ray / vLLM on the same Slurm allocation.
_DEFAULT_WORKERS = max(1, int(os.environ.get("APPS_REWARD_WORKERS", "32")))


def _parse_tests(ground_truth: Any) -> Optional[dict[str, Any]]:
    if ground_truth is None:
        return None
    if isinstance(ground_truth, dict):
        return ground_truth
    text = str(ground_truth).strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _all_pass(results: Any) -> bool:
    if not isinstance(results, list) or not results:
        return False
    return all(x is True for x in results)


def _pass_rate(results: Any) -> float:
    if not isinstance(results, list) or not results:
        return 0.0
    return sum(1 for x in results if x is True) / float(len(results))


def _reward_dict(
    *,
    score: float,
    acc: bool,
    pass_rate: float = 0.0,
    pred: str = "",
    extract: str = "none",
    error: str | None = None,
    n_tests: int = 0,
) -> dict[str, Any]:
    """Always emit the same keys — BatchRewardManager concatenates per-key lists."""
    return {
        "score": float(score),
        "acc": bool(acc),
        "pass_rate": float(pass_rate),
        "pred": pred,
        "extract": extract,
        "n_tests": int(n_tests),
        "error": error,
    }


def _score_one(
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
    *,
    timeout: int = 6,
    extract_profile: str = "strict",
) -> dict[str, Any]:
    tests = _parse_tests(ground_truth)
    if not tests or not (tests.get("inputs") and tests.get("outputs")):
        return _reward_dict(score=-1.0, acc=False, extract="none", error="missing_tests")

    code = extract_lcb_candidate(solution_str or "", profile=extract_profile)
    if not (code or "").strip():
        return _reward_dict(score=-1.0, acc=False, extract="empty", error="no_python_fence")

    if isinstance(extra_info, dict) and extra_info.get("timeout") is not None:
        try:
            timeout = int(extra_info["timeout"])
        except (TypeError, ValueError):
            pass

    sample = {"input_output": json.dumps(tests, ensure_ascii=False)}
    try:
        results, metadata = check_correctness(
            sample, code, timeout=int(timeout), debug=False
        )
    except Exception as exc:  # noqa: BLE001
        return _reward_dict(
            score=-1.0,
            acc=False,
            pred=code[:200],
            extract="python_fence",
            error=f"judge_exception:{type(exc).__name__}",
        )

    ok = _all_pass(results)
    rate = _pass_rate(results)
    err = None
    if isinstance(metadata, dict):
        err = metadata.get("error_message") or metadata.get("error")
    return _reward_dict(
        score=1.0 if ok else -1.0,
        acc=bool(ok),
        pass_rate=float(rate),
        pred=code[:200],
        extract="python_fence",
        n_tests=int(len(results) if isinstance(results, list) else 0),
        error=None if ok else (str(err) if err else "tests_failed"),
    )


def _score_batch(
    solution_strs: list[str],
    ground_truths: list[Any],
    extra_infos: list[Any] | None,
    *,
    timeout: int = 6,
    extract_profile: str = "strict",
    num_workers: int | None = None,
) -> list[dict[str, Any]]:
    n = len(solution_strs)
    extras = list(extra_infos) if extra_infos is not None else [None] * n
    if len(extras) != n:
        extras = (extras + [None] * n)[:n]
    workers = int(num_workers or _DEFAULT_WORKERS)
    workers = max(1, min(workers, n))

    out: list[Optional[dict[str, Any]]] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _score_one,
                solution_strs[i],
                ground_truths[i],
                extras[i],
                timeout=timeout,
                extract_profile=extract_profile,
            ): i
            for i in range(n)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                out[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                out[i] = _reward_dict(
                    score=-1.0,
                    acc=False,
                    extract="none",
                    error=f"pool_exception:{type(exc).__name__}",
                )
    return [
        x if x is not None else _reward_dict(score=-1.0, acc=False, error="missing_result")
        for x in out
    ]


def compute_score(
    data_source: str | None = None,
    solution_str: str = "",
    ground_truth: Any = "",
    extra_info: Any = None,
    *,
    # BatchRewardManager API
    data_sources: list[str] | None = None,
    solution_strs: list[str] | None = None,
    ground_truths: list[Any] | None = None,
    extra_infos: list[Any] | None = None,
    timeout: int = 6,
    extract_profile: str = "strict",
    num_workers: int | None = None,
    **kwargs: Any,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Score one completion or a batch (BatchRewardManager)."""
    _ = data_source, kwargs
    # Ignore DAPO-only kwargs when called via batch manager.
    kwargs.pop("max_resp_len", None)
    kwargs.pop("overlong_buffer_cfg", None)

    if solution_strs is not None:
        gts = ground_truths if ground_truths is not None else [""] * len(solution_strs)
        workers = num_workers
        if workers is None and isinstance(kwargs.get("apps_reward_workers"), (int, float)):
            workers = int(kwargs["apps_reward_workers"])
        return _score_batch(
            list(solution_strs),
            list(gts),
            list(extra_infos) if extra_infos is not None else None,
            timeout=timeout,
            extract_profile=extract_profile,
            num_workers=workers,
        )

    return _score_one(
        solution_str,
        ground_truth,
        extra_info,
        timeout=timeout,
        extract_profile=extract_profile,
    )
