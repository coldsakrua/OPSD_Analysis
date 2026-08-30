#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan eval_outputs for truncated think generations and pack SLURM shards."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

TARGET_MAX_NEW = 38912
DEFAULT_MAX_MODEL_LEN = 40960
DEFAULT_SHARD_TRUNC = 400


def _is_qwen3_family(dirname: str) -> bool:
    d = dirname.lower()
    if any(x in d for x in ("olmo", "deepseek", "falcon", "mimo", "r1_", "r1-")):
        return False
    if d.startswith(("qwen3", "qwen35", "s1p7", "sft_think", "st_", "snt_")):
        if any(x in d for x in ("olmo", "r1", "15b")):
            return False
        return True
    return False


def _pos_capped(dirname: str) -> bool:
    if dirname in ("qwen3-4b-base", "qwen3-1.7b-base"):
        return True
    if dirname.startswith(("st_tt_4bb", "st_tt_4bs", "st_tt_1p7bb", "st_tt_1p7bs")):
        return True
    return False


def _results_path(metrics_path: Path) -> Path:
    stem = metrics_path.name
    if stem.endswith(".metrics.json"):
        return metrics_path.with_name(stem[: -len(".metrics.json")] + ".results.json")
    return metrics_path.with_suffix("").with_suffix(".results.json")


def _count_truncations(results: Dict[str, Any], old_max: int, target: int) -> int:
    n = 0
    for row in results.get("results") or []:
        for g in row.get("generations") or []:
            if int(g.get("extended_from") or 0) > 0:
                continue
            t = int(g.get("num_tokens") or 0)
            if t >= old_max and t < target:
                n += 1
    return n


def _resolve_log(
    log_root: Path,
    metrics_path: Path,
    fallback_dir: Path,
    eval_tag: str,
    dataset: str,
) -> str:
    """Prefer original eval .out when cheap to find; else fallback extend log."""
    aliases = {
        "qwen3-0.6b": ["0.6b"],
        "qwen3-1.7b": ["1.7b"],
        "qwen3-1.7b_t10": ["1.7b"],
        "qwen3-4b": ["4b"],
        "qwen3-8b": ["8b"],
        "qwen3-4b-thinking": ["qwen3_4b_thinking"],
    }
    families = list(
        dict.fromkeys(
            aliases.get(eval_tag, [])
            + [
                eval_tag,
                eval_tag.replace("-", "_"),
                "1.7b" if ("1p7b" in eval_tag or "1.7b" in eval_tag) else "",
                "4b" if ("_4b" in eval_tag or eval_tag.endswith("4b") or "4b_" in eval_tag) else "",
                "8b" if "8b" in eval_tag else "",
                "0.6b" if ("06b" in eval_tag or "0.6b" in eval_tag) else "",
            ]
        )
    )
    families = [f for f in families if f]
    # Standard: fam/{dataset}/think; also lora/temperature and other nested layouts.
    search_dirs: List[Path] = []
    for fam in families:
        base = log_root / fam
        for p in (
            base / dataset / "think",
            base / "lora" / dataset / "think",
            base / "temperature" / dataset / "think",
            base / dataset / "lora" / "think",
        ):
            if p.is_dir():
                search_dirs.append(p)
        # Broad fallback: whole family tree (covers odd paths)
        if base.is_dir():
            search_dirs.append(base)

    needle = str(metrics_path.resolve())
    needle_alt = f"eval_outputs/{eval_tag}/"
    if shutil.which("rg") and search_dirs:
        seen: set = set()
        for d in search_dirs:
            key = str(d.resolve()) if d.exists() else str(d)
            if key in seen:
                continue
            seen.add(key)
            try:
                proc = subprocess.run(
                    [
                        "rg",
                        "-l",
                        "--glob",
                        "*.out",
                        "--max-count",
                        "1",
                        "-F",
                        needle,
                        str(d),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
            except Exception:
                continue
            hits = [Path(x) for x in proc.stdout.splitlines() if x.strip()]
            if not hits:
                try:
                    proc2 = subprocess.run(
                        [
                            "rg",
                            "-l",
                            "--glob",
                            "*.out",
                            "--max-count",
                            "1",
                            "-F",
                            needle_alt,
                            str(d),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    hits = [Path(x) for x in proc2.stdout.splitlines() if x.strip()]
                except Exception:
                    hits = []
            # Drop extend_38912 fallbacks if we somehow searched there
            hits = [p for p in hits if "extend_38912" not in str(p)]
            if hits:
                hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return str(hits[0].resolve())

    fb = fallback_dir / eval_tag / f"{dataset}_extend.out"
    fb.parent.mkdir(parents=True, exist_ok=True)
    return str(fb)


def _infer_dataset(metrics: Dict[str, Any], metrics_path: Path) -> str:
    args = metrics.get("dataset_args") or []
    if args:
        return str(args[0])
    name = metrics_path.name
    for ds in ("aime24", "aime25", "aime26", "hmmt25", "math500"):
        if name.startswith(ds + "_"):
            return ds
    return "unknown"


def scan_jobs(
    eval_outputs: Path,
    log_root: Path,
    fallback_log_dir: Path,
    target: int,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for metrics_path in sorted(eval_outputs.glob("*/*think*.metrics.json")):
        if "nothink" in metrics_path.name.lower():
            continue
        eval_tag = metrics_path.parent.name
        if not _is_qwen3_family(eval_tag) or "smoke" in eval_tag or _pos_capped(eval_tag):
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        old_max = int(metrics.get("max_new_tokens") or 0)
        if old_max <= 0:
            continue
        already_extended = int(metrics.get("extended_to") or 0) >= target
        if old_max >= target and not already_extended:
            # e.g. 4b-thinking at 81920 — nothing to do
            continue
        if already_extended:
            old_max = int(metrics.get("pre_extend_max_new_tokens") or 32768)
        results_path = _results_path(metrics_path)
        if not results_path.is_file():
            continue
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        trunc = _count_truncations(results, old_max, target)
        if trunc <= 0:
            continue
        model_path = metrics.get("model_path") or metrics.get("vllm_base_model_path") or ""
        if not model_path:
            continue
        dataset = _infer_dataset(metrics, metrics_path)
        log_out = _resolve_log(log_root, metrics_path, fallback_log_dir, eval_tag, dataset)
        jobs.append(
            {
                "eval_tag": eval_tag,
                "dataset": dataset,
                "metrics_json": str(metrics_path.resolve()),
                "results_json": str(results_path.resolve()),
                "model_path": model_path,
                "vllm_base_model_path": metrics.get("vllm_base_model_path") or model_path,
                "lora_adapter_dir": metrics.get("lora_adapter_dir"),
                "checkpoint_dir": metrics.get("checkpoint_dir"),
                "enable_thinking": bool(metrics.get("enable_thinking", True)),
                "temperature": float(metrics.get("temperature", 0.6)),
                "top_p": float(metrics.get("top_p", 0.95)),
                "top_k": int(metrics.get("top_k", 20)),
                "min_p": float(metrics.get("min_p", 0.0)),
                "presence_penalty": float(metrics.get("presence_penalty", 0.0)),
                "seed": 42,
                "old_max_new": old_max,
                "target_max_new": target,
                "max_model_len": DEFAULT_MAX_MODEL_LEN,
                "trunc_count": trunc,
                "pass_at_k_list": metrics.get("pass_at_k_list") or [1, 4, 8],
                "relaxed_answer_extraction": bool(metrics.get("relaxed_answer_extraction", False)),
                "log_out": log_out,
            }
        )
    return jobs


def pack_shards(jobs: List[Dict[str, Any]], shard_trunc: int) -> List[List[Dict[str, Any]]]:
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for j in jobs:
        by_model[j["model_path"]].append(j)
    model_items = sorted(
        by_model.items(),
        key=lambda kv: -sum(x["trunc_count"] for x in kv[1]),
    )
    shards: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_trunc = 0
    for _model, group in model_items:
        group_trunc = sum(x["trunc_count"] for x in group)
        if cur and cur_trunc + group_trunc > shard_trunc and cur_trunc > 0:
            shards.append(cur)
            cur = []
            cur_trunc = 0
        if group_trunc > shard_trunc * 2 and len(group) > 1:
            for j in group:
                if cur and cur_trunc + j["trunc_count"] > shard_trunc and cur_trunc > 0:
                    shards.append(cur)
                    cur = []
                    cur_trunc = 0
                cur.append(j)
                cur_trunc += j["trunc_count"]
        else:
            cur.extend(group)
            cur_trunc += group_trunc
    if cur:
        shards.append(cur)
    return shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3]),
    )
    parser.add_argument("--target-max-new", type=int, default=TARGET_MAX_NEW)
    parser.add_argument("--shard-trunc", type=int, default=DEFAULT_SHARD_TRUNC)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = Path(args.base_dir).expanduser().resolve()
    eval_outputs = base / "eval_outputs"
    log_root = base / "log" / "eval"
    out_dir = base / "scripts" / "eval" / "extend_38912" / "manifest"
    shard_dir = out_dir / "shards"
    fallback_log = base / "log" / "eval" / "extend_38912"

    jobs = scan_jobs(eval_outputs, log_root, fallback_log, args.target_max_new)
    shards = pack_shards(jobs, args.shard_trunc)

    print(
        f"[manifest] jobs={len(jobs)} trunc_gens={sum(j['trunc_count'] for j in jobs)} "
        f"dirs={len({j['eval_tag'] for j in jobs})} shards={len(shards)}"
    )
    for i, sh in enumerate(shards):
        print(
            f"  shard_{i:02d}: jobs={len(sh)} trunc={sum(x['trunc_count'] for x in sh)} "
            f"models={len({x['model_path'] for x in sh})}"
        )

    if args.dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob("shard_*.jsonl"):
        old.unlink()

    man_path = out_dir / "manifest.jsonl"
    with man_path.open("w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(f"[manifest] wrote {man_path}")

    for i, sh in enumerate(shards):
        sp = shard_dir / f"shard_{i:02d}.jsonl"
        with sp.open("w", encoding="utf-8") as f:
            for j in sh:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
        print(f"[manifest] wrote {sp}")

    summary = {
        "jobs": len(jobs),
        "trunc_gens": sum(j["trunc_count"] for j in jobs),
        "dirs": sorted({j["eval_tag"] for j in jobs}),
        "n_shards": len(shards),
        "shard_trunc": args.shard_trunc,
        "target_max_new": args.target_max_new,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
