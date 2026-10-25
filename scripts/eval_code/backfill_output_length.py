#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backfill avg_output_tokens for finished code-eval runs; append to matching logs."""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _metrics_path_for_gens(gens: Path) -> Path:
    return Path(str(gens).replace(".generations.json", ".json"))


def load_generation_rows(path: Path) -> List[Dict[str, Any]]:
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


def summarize_output_lengths(counts: Sequence[int]) -> Dict[str, Any]:
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


def _resolve_model_path(metrics: Dict[str, Any], metrics_path: Path) -> Optional[str]:
    mp = metrics.get("model_path")
    if isinstance(mp, str) and mp:
        return mp
    for sib in sorted(metrics_path.parent.glob("*.json")):
        name = sib.name
        if any(
            x in name
            for x in (
                ".generations.",
                ".results.",
                ".samples.",
                ".subset.",
                "eval_results",
            )
        ):
            continue
        if sib == metrics_path:
            continue
        try:
            doc = json.loads(sib.read_text(encoding="utf-8"))
        except Exception:
            continue
        sib_mp = doc.get("model_path")
        if isinstance(sib_mp, str) and sib_mp:
            return sib_mp
    return None


def _find_logs_for_output(metrics_path: Path, gens_path: Path, log_root: Path) -> List[Path]:
    hits: List[Path] = []
    parent = metrics_path.parent.name
    stem = metrics_path.stem
    for log in log_root.rglob("*.out"):
        try:
            text = log.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if (
            f"output={metrics_path}" in text
            or f"wrote generations → {gens_path}" in text
            or f"wrote metrics → {metrics_path}" in text
            or f"generations={gens_path}" in text
        ):
            hits.append(log)
            continue
        if parent and parent in text and stem in text:
            hits.append(log)
    return sorted(set(hits))


def _log_prefix(log_path: Path, text: str) -> str:
    name = log_path.name
    head = text[:4000]
    if "_j." in name or "[evalplus-judge]" in head or "[lcb-judge]" in head:
        if "[lcb-judge]" in head or "lcb_" in name:
            return "[lcb-judge]"
        return "[evalplus-judge]"
    if "[lcb]" in head or "/lcb_v6/" in str(log_path) or name.startswith("lcb_"):
        return "[lcb]"
    return "[evalplus]"


def _format_line(prefix: str, summary: Dict[str, Any]) -> str:
    return (
        f"{prefix} avg_output_tokens={summary['avg_tokens']:.1f} "
        f"(n={summary['n_samples']} median={summary['median_tokens']:.1f} "
        f"min={summary['min_tokens']} max={summary['max_tokens']})"
    )


def _append_log_line(log_path: Path, line: str) -> None:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    if "avg_output_tokens=" in text:
        lines = text.splitlines()
        new_lines = []
        replaced = False
        for ln in lines:
            if re.search(r"avg_output_tokens=", ln) and "Processed prompts" not in ln:
                new_lines.append(line)
                replaced = True
            else:
                new_lines.append(ln)
        if replaced:
            ending = "\n" if text.endswith("\n") else ""
            log_path.write_text("\n".join(new_lines) + ending, encoding="utf-8")
            return
    with log_path.open("a", encoding="utf-8") as f:
        if text and not text.endswith("\n"):
            f.write("\n")
        f.write(line + "\n")


def _count_tokens(rows: List[Dict[str, Any]], tokenizer: Any) -> List[int]:
    counts: List[int] = []
    for row in rows:
        stored = row.get("num_tokens_list")
        gens = row.get("output_list") or []
        if isinstance(stored, list) and stored and (not gens or len(stored) == len(gens)):
            counts.extend(int(x) for x in stored)
            continue
        for g in gens:
            t = g if isinstance(g, str) else ""
            counts.append(len(tokenizer.encode(t, add_special_tokens=False)) if t else 0)
    return counts


def process_one(
    gens_path: Path,
    *,
    log_root: Path,
    tok_cache: Dict[str, Any],
    force: bool,
    dry_run: bool,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    metrics_path = _metrics_path_for_gens(gens_path)
    if not metrics_path.is_file():
        return f"SKIP no metrics: {gens_path}", None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"SKIP bad metrics {metrics_path}: {e}", None

    cached = (
        not force
        and metrics.get("avg_output_tokens_mean") is not None
        and isinstance(metrics.get("output_length"), dict)
        and metrics["output_length"].get("n_samples")
    )
    if cached:
        summary = metrics["output_length"]
        logs = _find_logs_for_output(metrics_path, gens_path, log_root)
        if not dry_run:
            for lp in logs:
                prefix = _log_prefix(lp, lp.read_text(encoding="utf-8", errors="ignore"))
                _append_log_line(lp, _format_line(prefix, summary))
        return (
            f"OK cached {gens_path.parent.name}/{gens_path.name} "
            f"avg={summary['avg_tokens']:.1f} logs={len(logs)}",
            summary,
        )

    model_path = _resolve_model_path(metrics, metrics_path)
    if not model_path:
        return f"SKIP no model_path: {metrics_path}", None

    if model_path not in tok_cache:
        print(f"[backfill] load tokenizer: {model_path}", flush=True)
        tok_cache[model_path] = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
    tokenizer = tok_cache[model_path]

    print(f"[backfill] tokenize {gens_path}", flush=True)
    rows = load_generation_rows(gens_path)
    if not rows:
        return f"SKIP empty gens: {gens_path}", None

    counts = _count_tokens(rows, tokenizer)
    summary = summarize_output_lengths(counts)
    if dry_run:
        return (
            f"DRY {gens_path.parent.name}/{gens_path.name} avg={summary['avg_tokens']:.1f}",
            summary,
        )

    metrics["avg_output_tokens_mean"] = float(summary["avg_tokens"])
    metrics["output_length"] = summary
    if not metrics.get("model_path"):
        metrics["model_path"] = model_path
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    results_path = metrics_path.with_suffix(".results.json")
    if results_path.is_file():
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
            if isinstance(results, dict):
                results["avg_output_tokens_mean"] = float(summary["avg_tokens"])
                results["output_length"] = summary
                results_path.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        except Exception:
            pass

    logs = _find_logs_for_output(metrics_path, gens_path, log_root)
    for lp in logs:
        prefix = _log_prefix(lp, lp.read_text(encoding="utf-8", errors="ignore"))
        _append_log_line(lp, _format_line(prefix, summary))

    return (
        f"OK {gens_path.parent.name}/{gens_path.name} "
        f"avg={summary['avg_tokens']:.1f} n={summary['n_samples']} logs={len(logs)}",
        summary,
    )


def _worker_process_one(payload: Tuple[str, str, bool, bool]) -> str:
    gens_s, log_root_s, force, dry_run = payload
    # Each worker keeps its own tokenizer cache.
    msg, _ = process_one(
        Path(gens_s),
        log_root=Path(log_root_s),
        tok_cache={},
        force=force,
        dry_run=dry_run,
    )
    return msg


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill code-eval avg_output_tokens + logs")
    p.add_argument("--eval-outputs", type=str, default=str(_repo() / "eval_outputs"))
    p.add_argument("--log-root", type=str, default=str(_repo() / "log" / "eval_code"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", type=str, default="", help="Substring filter on generations path")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers (0 = SLURM_CPUS_PER_TASK or 1)",
    )
    args = p.parse_args()

    eval_root = Path(args.eval_outputs)
    log_root = Path(args.log_root)
    gens_list = sorted(eval_root.rglob("*.generations.json"))
    if args.only:
        gens_list = [g for g in gens_list if args.only in str(g)]

    workers = int(args.workers)
    if workers <= 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK") or 1)
    workers = max(1, min(workers, len(gens_list) or 1))
    print(f"[backfill] files={len(gens_list)} workers={workers}", flush=True)

    ok = 0
    skip = 0
    if workers == 1:
        tok_cache: Dict[str, Any] = {}
        for g in gens_list:
            msg, _summary = process_one(
                g,
                log_root=log_root,
                tok_cache=tok_cache,
                force=bool(args.force),
                dry_run=bool(args.dry_run),
            )
            print(msg, flush=True)
            if msg.startswith("OK") or msg.startswith("DRY"):
                ok += 1
            else:
                skip += 1
    else:
        payloads = [
            (str(g), str(log_root), bool(args.force), bool(args.dry_run))
            for g in gens_list
        ]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker_process_one, pl) for pl in payloads]
            for fut in as_completed(futs):
                msg = fut.result()
                print(msg, flush=True)
                if msg.startswith("OK") or msg.startswith("DRY"):
                    ok += 1
                else:
                    skip += 1
    print(f"[backfill] done ok={ok} skip={skip} total={len(gens_list)}", flush=True)


if __name__ == "__main__":
    main()
