#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for OPSD HumanEval+ / MBPP+ eval.

Prompting follows official EvalPlus instruct prefixes
(evalplus.codegen / provider.utility.make_raw_chat_prompt).
Judging calls the vendored official ``evalplus.evaluate``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.eval import PASS, estimate_pass_at_k
from evalplus.evaluate import evaluate as evalplus_evaluate
from evalplus.provider.utility import (
    _MAGIC_SPLITTER_,
    make_raw_chat_prompt,
)
from evalplus.sanitize import sanitize
from code_extract import PROFILE_STRICT, extract_evalplus_candidate

# Skip official O(n^2) longest-span search on dumps this long; treat as extract miss.
_MAX_SANITIZE_LINES = 400

# Official EvalPlus instruction / response prefixes (codegen.py).
INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
RESPONSE_PREFIX = (
    "Below is a Python script with a self-contained function that solves the "
    "problem and passes corresponding tests:"
)

DATASET_ALIASES = {
    "humaneval": "humaneval",
    "humanevalplus": "humaneval",
    "humaneval+": "humaneval",
    "he": "humaneval",
    "mbpp": "mbpp",
    "mbppplus": "mbpp",
    "mbpp+": "mbpp",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_dataset(name: str) -> str:
    key = name.strip().lower().replace("_", "").replace("-", "")
    if key not in DATASET_ALIASES:
        raise ValueError(
            f"Unknown dataset {name!r}; expected humaneval(+)/mbpp(+)"
        )
    return DATASET_ALIASES[key]


def default_data_path(dataset: str) -> Path:
    root = repo_root()
    ds = normalize_dataset(dataset)
    if ds == "humaneval":
        return root / "data" / "humanevalplus" / "HumanEvalPlus.jsonl"
    return root / "data" / "mbppplus" / "MbppPlus.jsonl"


def configure_evalplus_data(dataset: str, data_jsonl: Optional[str] = None) -> Path:
    """Point official EvalPlus loaders at local HumanEvalPlus / MbppPlus jsonl."""
    ds = normalize_dataset(dataset)
    path = Path(data_jsonl) if data_jsonl else default_data_path(ds)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing official EvalPlus jsonl for {ds}: {path}\n"
            f"Expected HumanEvalPlus.jsonl or MbppPlus.jsonl "
            f"(with base_input / plus_input / entry_point)."
        )
    if ds == "humaneval":
        os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(path.resolve())
    else:
        os.environ["MBPP_OVERRIDE_PATH"] = str(path.resolve())
    return path.resolve()


def load_problems(dataset: str) -> Dict[str, Dict[str, Any]]:
    ds = normalize_dataset(dataset)
    if ds == "humaneval":
        return get_human_eval_plus()
    return get_mbpp_plus()


def write_subset_override_jsonl(
    dataset: str,
    problems: Sequence[Dict[str, Any]],
    dest: Path,
) -> Path:
    """Write a filtered HumanEvalPlus/MbppPlus jsonl and retarget OVERRIDE_PATH.

    Official ``evalplus.evaluate`` asserts samples cover *all* loaded problems, so
    ``--max-problems`` / debug subsets must shrink the dataset seen by the judge.
    """

    def _jsonable(x: Any) -> Any:
        if isinstance(x, tuple):
            return [_jsonable(v) for v in x]
        if isinstance(x, list):
            return [_jsonable(v) for v in x]
        if isinstance(x, dict):
            return {k: _jsonable(v) for k, v in x.items()}
        return x

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for task in problems:
            row = {
                k: _jsonable(v)
                for k, v in task.items()
                if not str(k).startswith("_")
            }
            f.write(json.dumps(row) + "\n")
    configure_evalplus_data(dataset, str(dest))
    return dest.resolve()

def _apply_chat_template(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    enable_thinking: bool,
    add_generation_prompt: bool,
) -> str:
    kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        kwargs["enable_thinking"] = bool(enable_thinking)
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def ensure_think_open(prompt: str) -> str:
    """Make the assistant turn start a think block.

    Qwen3 chat templates already append ``<think>`` when
    ``enable_thinking=True`` + ``add_generation_prompt=True``. Olmo often does
    not; append it manually if the suffix is missing.
    """
    text = prompt or ""
    if "<think>" in text[-240:]:
        return text if text.endswith("\n") else text + "\n"
    return text.rstrip() + "\n<think>\n"


def overlay_chat_template(tokenizer: Any, template_src: Optional[str]) -> None:
    """Replace tokenizer.chat_template from an instruct/think checkpoint.

    Qwen3-Base ships a slightly different template than Qwen3; training overlays
    the instruct template. Eval should do the same when ``template_src`` is set.
    """
    if not template_src:
        return
    src = Path(template_src)
    if src.is_dir():
        cfg_path = src / "tokenizer_config.json"
    else:
        cfg_path = src
    if not cfg_path.is_file():
        raise FileNotFoundError(f"chat template source missing: {cfg_path}")
    if cfg_path.suffix == ".json":
        tmpl = json.loads(cfg_path.read_text()).get("chat_template")
    else:
        tmpl = cfg_path.read_text()
    if not tmpl:
        raise ValueError(f"no chat_template in {cfg_path}")
    tokenizer.chat_template = tmpl
    print(f"[prompt] overlay chat_template from {cfg_path}")


def build_prompt(
    tokenizer: Any,
    task: Dict[str, Any],
    *,
    enable_thinking: bool,
    force_base_prompt: bool = False,
    prefill_think: bool = True,
) -> str:
    """Build one generation prompt.

    - Base / completion models: raw ``task['prompt']`` (official direct completion).
    - Chat + nothink: official ``make_raw_chat_prompt`` (assistant prefills ```python).
    - Chat + think: official template with ``enable_thinking`` and generation
      prompt (ends in ``<think>``). Do not prefill `````python`` — that skips
      thinking. Olmo gets ``<think>`` appended if the template omitted it.
    """
    task_prompt = task["prompt"].strip() + "\n"
    if force_base_prompt or getattr(tokenizer, "chat_template", None) is None:
        return task_prompt

    if not enable_thinking:
        return make_raw_chat_prompt(
            task_prompt,
            INSTRUCTION_PREFIX,
            RESPONSE_PREFIX,
            tokenizer,
        )

    user = (
        f"{INSTRUCTION_PREFIX}\n"
        f"```\n{task_prompt.strip()}\n```\n"
    )
    text = _apply_chat_template(
        tokenizer,
        [{"role": "user", "content": user}],
        enable_thinking=True,
        add_generation_prompt=True,
    )
    if prefill_think:
        return ensure_think_open(text)
    return text


def stop_token_ids(tokenizer: Any) -> List[int]:
    ids: List[int] = []
    for tok in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0 and tid not in ids:
                ids.append(tid)
        except Exception:
            pass
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int) and eos not in ids:
        ids.append(eos)
    elif isinstance(eos, list):
        for x in eos:
            if isinstance(x, int) and x not in ids:
                ids.append(x)
    return ids


def _code_candidate_for_sanitize(
    solution: str,
    entrypoint: Optional[str] = None,
    profile: str = PROFILE_STRICT,
) -> str:
    """Shrink LLM dump to a code-sized snippet before EvalPlus sanitize."""
    return extract_evalplus_candidate(
        solution, entrypoint=entrypoint, profile=profile
    )


def sanitize_generations(
    problems: Sequence[Dict[str, Any]],
    raw_outputs: Sequence[Sequence[str]],
    *,
    is_direct_completion: bool,
    extract_profile: str = PROFILE_STRICT,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Return (raw_solutions, sanitized_solutions) aligned with problems × samples."""
    raw_sols: List[List[str]] = []
    san_sols: List[List[str]] = []
    for task, gens in zip(problems, raw_outputs):
        prompt = task["prompt"].strip() + "\n"
        entry = task["entry_point"]
        raw_row: List[str] = []
        san_row: List[str] = []
        for gen in gens:
            gen = (gen or "").replace("\t", "    ")
            solution = prompt + gen if is_direct_completion else gen
            raw_row.append(solution)
            candidate = _code_candidate_for_sanitize(
                solution, entrypoint=entry, profile=extract_profile
            )
            if is_direct_completion and not candidate.startswith(prompt.strip()[:40]):
                candidate = prompt + candidate
            nlines = candidate.count("\n") + 1 if candidate else 0
            if not (candidate or "").strip() or nlines > _MAX_SANITIZE_LINES:
                san_row.append("")
            else:
                san_row.append(sanitize(candidate, entrypoint=entry))
        raw_sols.append(raw_row)
        san_sols.append(san_row)
        if len(raw_sols) == 1 or len(raw_sols) % 50 == 0:
            print(
                f"[evalplus] sanitize {len(raw_sols)}/{len(problems)}",
                flush=True,
            )
    return raw_sols, san_sols


def write_samples_jsonl(
    path: Path,
    problems: Sequence[Dict[str, Any]],
    solutions: Sequence[Sequence[str]],
) -> Path:
    """Official EvalPlus samples format: one jsonl line per (task, sample)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task, sols in zip(problems, solutions):
            tid = task["task_id"]
            for sol in sols:
                f.write(json.dumps({"task_id": tid, "solution": sol}) + "\n")
    return path


def append_jsonl_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def append_samples_jsonl(
    path: Path,
    problems: Sequence[Dict[str, Any]],
    solutions: Sequence[Sequence[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for task, sols in zip(problems, solutions):
            tid = task["task_id"]
            for sol in sols:
                f.write(json.dumps({"task_id": tid, "solution": sol}) + "\n")
        f.flush()


def dump_generations_json(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Compact JSON array for the CPU judge (no indent: huge CoT dumps are expensive)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), ensure_ascii=False))


def load_generation_rows(path: Path) -> List[Dict[str, Any]]:
    """Accept a JSON array or JSONL checkpoint."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise SystemExit(f"{path} is not a JSON list")
        return data
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _status_pass(status: Optional[str]) -> bool:
    return status == PASS


def compute_pass_at_k_from_eval_results(
    eval_results: Dict[str, Any],
    k_list: Sequence[int],
    *,
    base_only: bool = False,
) -> Dict[str, Any]:
    """Official unbiased pass@k on base / plus from evalplus eval_results.json."""
    eval_map: Dict[str, List[Dict[str, Any]]] = eval_results.get("eval", {})
    if not eval_map:
        return {"base": {}, "plus": {}}

    total = np.array([len(rows) for rows in eval_map.values()], dtype=int)
    base_correct = []
    plus_correct = []
    for rows in eval_map.values():
        base_correct.append(sum(_status_pass(r.get("base_status")) for r in rows))
        if not base_only:
            plus_correct.append(
                sum(
                    _status_pass(r.get("base_status")) and _status_pass(r.get("plus_status"))
                    for r in rows
                )
            )
    base_correct_arr = np.array(base_correct, dtype=int)
    out: Dict[str, Any] = {"base": {}, "plus": {}}
    for k in k_list:
        if int(total.min()) < k:
            continue
        out["base"][f"pass@{k}"] = float(
            estimate_pass_at_k(total, base_correct_arr, k).mean()
        )
    if not base_only and plus_correct:
        plus_correct_arr = np.array(plus_correct, dtype=int)
        for k in k_list:
            if not (total >= k).all():
                continue
            out["plus"][f"pass@{k}"] = float(
                estimate_pass_at_k(total, plus_correct_arr, k).mean()
            )
    return out


def compute_math_style_pass_at_k(
    eval_results: Dict[str, Any],
    k_list: Sequence[int],
    *,
    base_only: bool = False,
) -> Dict[str, Any]:
    """Any-of-first-k pass rate (same reporting style as OPSD LCB math-style)."""
    eval_map: Dict[str, List[Dict[str, Any]]] = eval_results.get("eval", {})
    n = len(eval_map)
    base_counts = {k: 0 for k in k_list}
    plus_counts = {k: 0 for k in k_list}
    per_problem: List[Dict[str, Any]] = []

    for task_id, rows in eval_map.items():
        base_flags = [_status_pass(r.get("base_status")) for r in rows]
        plus_flags = [
            _status_pass(r.get("base_status")) and _status_pass(r.get("plus_status"))
            for r in rows
        ]
        base_ok = {k: (len(base_flags) >= 1 and any(base_flags[:k])) for k in k_list}
        plus_ok = {k: (len(plus_flags) >= 1 and any(plus_flags[:k])) for k in k_list}
        for k in k_list:
            if base_ok[k]:
                base_counts[k] += 1
            if not base_only and plus_ok[k]:
                plus_counts[k] += 1
        per_problem.append(
            {
                "task_id": task_id,
                "base_graded": base_flags,
                "plus_graded": plus_flags,
                "pass_at_k_base": base_ok,
                "pass_at_k_plus": plus_ok,
            }
        )

    def _pack(counts: Dict[int, int]) -> Dict[str, Any]:
        return {
            str(k): {
                "n_solved": counts[k],
                "n_total": n,
                "rate": counts[k] / n if n else 0.0,
                "rate_pct": 100.0 * counts[k] / n if n else 0.0,
            }
            for k in k_list
        }

    return {
        "base": _pack(base_counts),
        "plus": {} if base_only else _pack(plus_counts),
        "problems": per_problem,
    }


def completion_token_count(
    text: str,
    token_ids: Optional[Sequence[int]],
    tokenizer: Any,
) -> int:
    """Completion length in tokens (prefer text encode when vLLM ids look padded)."""
    text_count = len(tokenizer.encode(text, add_special_tokens=False)) if text else 0
    if not token_ids:
        return text_count
    ids_count = len(token_ids)
    # vLLM can return padded/incorrect token_ids for some models (e.g. Gemma 3).
    if text and ids_count > max(text_count + 128, text_count * 2):
        return text_count
    return ids_count


def collect_output_token_counts(
    rows: Sequence[Dict[str, Any]],
    *,
    tokenizer: Any = None,
) -> List[int]:
    """Gather per-sample output token counts from generations rows.

    Prefers stored ``num_tokens_list``; otherwise encodes ``output_list`` when a
    tokenizer is provided.
    """
    counts: List[int] = []
    for row in rows:
        stored = row.get("num_tokens_list")
        gens = row.get("output_list") or []
        if isinstance(stored, list) and stored:
            if gens and len(stored) != len(gens):
                # Fall back to re-encode when lists are misaligned.
                stored = None
            else:
                counts.extend(int(x) for x in stored)
                continue
        if tokenizer is None:
            continue
        for g in gens:
            t = g if isinstance(g, str) else ""
            counts.append(
                len(tokenizer.encode(t, add_special_tokens=False)) if t else 0
            )
    return counts


def summarize_output_lengths(counts: Sequence[int]) -> Dict[str, Any]:
    """Aggregate stats aligned with math-eval ``avg_output_tokens_mean``."""
    if not counts:
        return {
            "n_samples": 0,
            "avg_tokens": 0.0,
            "median_tokens": 0.0,
            "min_tokens": 0,
            "max_tokens": 0,
        }
    sorted_c = sorted(int(x) for x in counts)
    n = len(sorted_c)
    mid = n // 2
    median = (
        float(sorted_c[mid])
        if n % 2 == 1
        else 0.5 * (sorted_c[mid - 1] + sorted_c[mid])
    )
    return {
        "n_samples": n,
        "avg_tokens": float(sum(sorted_c) / n),
        "median_tokens": median,
        "min_tokens": int(sorted_c[0]),
        "max_tokens": int(sorted_c[-1]),
    }


def attach_output_length_metrics(
    metrics_doc: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    *,
    tokenizer: Any = None,
) -> Dict[str, Any]:
    """Write ``avg_output_tokens_mean`` (+ detail) into metrics_doc; return summary."""
    counts = collect_output_token_counts(rows, tokenizer=tokenizer)
    summary = summarize_output_lengths(counts)
    metrics_doc["avg_output_tokens_mean"] = float(summary["avg_tokens"])
    metrics_doc["output_length"] = summary
    return summary


def compute_avg_at_k_from_eval_results(
    eval_results: Dict[str, Any],
    k_list: Sequence[int],
    *,
    base_only: bool = False,
) -> Dict[str, Any]:
    """Mean correctness over first-k samples (math-eval avg{n}_pct style)."""
    eval_map: Dict[str, List[Dict[str, Any]]] = eval_results.get("eval", {})
    out: Dict[str, Any] = {"base": {}, "plus": {}}

    def _avg(flags_fn) -> Dict[str, Any]:
        packed: Dict[str, Any] = {}
        for k in k_list:
            n_correct = 0
            n_total = 0
            for rows in eval_map.values():
                flags = flags_fn(rows)[:k]
                n_correct += sum(1 for x in flags if x)
                n_total += len(flags)
            rate = (n_correct / n_total) if n_total else 0.0
            packed[str(k)] = {
                "n_correct": n_correct,
                "n_total": n_total,
                "rate": rate,
                "rate_pct": 100.0 * rate,
            }
        return packed

    out["base"] = _avg(
        lambda rows: [_status_pass(r.get("base_status")) for r in rows]
    )
    if not base_only:
        out["plus"] = _avg(
            lambda rows: [
                _status_pass(r.get("base_status")) and _status_pass(r.get("plus_status"))
                for r in rows
            ]
        )
    return out


def run_official_evaluate(
    dataset: str,
    samples_jsonl: Path,
    *,
    parallel: Optional[int] = None,
    base_only: bool = False,
    mini: bool = False,
    noextreme: bool = False,
    version: str = "default",
    i_just_wanna_run: bool = True,
) -> Path:
    """Invoke official ``evalplus.evaluate``; return path to eval_results json."""
    ds = normalize_dataset(dataset)
    samples_jsonl = samples_jsonl.resolve()
    result_path = Path(str(samples_jsonl).replace(".jsonl", ".eval_results.json"))
    if result_path.is_file() and i_just_wanna_run:
        result_path.unlink()

    evalplus_evaluate(
        dataset=ds,
        samples=str(samples_jsonl),
        base_only=base_only,
        parallel=parallel,
        i_just_wanna_run=False,
        test_details=False,
        mini=mini,
        noextreme=noextreme,
        version=version,
        output_file=str(result_path),
    )
    if not result_path.is_file():
        # Older naming fallback used inside evaluate().
        legacy = Path(str(samples_jsonl).replace(".jsonl", "_eval_results.json"))
        if legacy.is_file():
            return legacy
        raise FileNotFoundError(f"EvalPlus did not write results at {result_path}")
    return result_path
