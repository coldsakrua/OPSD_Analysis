#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build extend-to-38912 shards for qwen3_*-base family (max_pos=32768).

These were excluded from the main manifest because config max_position_embeddings
is 32768. Extending to 38912 requires VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 and a
truncation slack (gens stop slightly below max_new due to prompt tokens).

One model per shard to avoid multi-model GPU teardown OOM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

TARGET_MAX_NEW = 38912
DEFAULT_MAX_MODEL_LEN = 40960
TRUNC_SLACK = 512  # treat n >= old_max - slack as context-truncated

# Tags under log/eval/qwen3_{1.7b,4b}_base (non-smoke, full 4-dataset runs).
DEFAULT_TAGS = [
    "qwen3-1.7b-base",
    "qwen3-4b-base",
    "st_tt_1p7bb_ckpt100",
    "st_tt_1p7bs5k_ckpt100",
    "st_tt_1p7bs10k_ckpt100",
    "st_tt_1p7bs15k_ckpt100",
    "st_tt_4bb_ckpt100",
    "st_tt_4bs5k_ckpt100",
    "st_tt_4bs10k_ckpt100",
    "st_tt_4bs15k_ckpt100",
    "sft_think_4gpu_12k_ckpt5000",
    "sft_think_4gpu_12k_ckpt7500",
    "sft_think_4gpu_12k_ckpt10000",
    "sft_think_4gpu_12k_ckpt12500",
    "sft_think_4gpu_12k_ckpt15000",
]


def _results_path(metrics_path: Path) -> Path:
    stem = metrics_path.name
    if stem.endswith(".metrics.json"):
        return metrics_path.with_name(stem[: -len(".metrics.json")] + ".results.json")
    return metrics_path.with_suffix("").with_suffix(".results.json")


def _infer_dataset(metrics: Dict[str, Any], metrics_path: Path) -> str:
    args = metrics.get("dataset_args") or []
    if args:
        return str(args[0])
    name = metrics_path.name
    for ds in ("aime24", "aime25", "aime26", "hmmt25", "math500"):
        if name.startswith(ds + "_"):
            return ds
    return "unknown"


def _count_truncations(results: Dict[str, Any], old_max: int, target: int, slack: int) -> int:
    thr = max(0, old_max - slack)
    n = 0
    for row in results.get("results") or []:
        for g in row.get("generations") or []:
            if int(g.get("extended_from") or 0) > 0:
                continue
            t = int(g.get("num_tokens") or 0)
            if t >= thr and t < target:
                n += 1
    return n


def _resolve_model_path(raw: str, base_dir: Path) -> Optional[str]:
    if not raw:
        return None
    p = Path(raw)
    if p.is_file() or (p.is_dir() and (p / "config.json").is_file()):
        return str(p.resolve())
    cand = (base_dir / raw).resolve()
    if cand.is_dir() and (cand / "config.json").is_file():
        return str(cand)
    return None


def _log_out_for(base_dir: Path, eval_tag: str, dataset: str, metrics_path: Path) -> str:
    """Prefer existing eval .out under qwen3_*_base; else fallback extend log."""
    import shutil
    import subprocess

    log_root = base_dir / "log" / "eval"
    families = []
    if "4b" in eval_tag or eval_tag.endswith("4b-base"):
        families.append("qwen3_4b_base")
    if "1p7" in eval_tag or "1.7b" in eval_tag or "sft_think" in eval_tag:
        families.append("qwen3_1.7b_base")
    families.extend(["qwen3_1.7b_base", "qwen3_4b_base"])

    needle = str(metrics_path.resolve())
    if shutil.which("rg"):
        for fam in dict.fromkeys(families):
            think_dir = log_root / fam / dataset / "think"
            if not think_dir.is_dir():
                continue
            try:
                proc = subprocess.run(
                    ["rg", "-l", "--glob", "*.out", "--max-count", "1", "-F", needle, str(think_dir)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except Exception:
                continue
            hits = [Path(x) for x in proc.stdout.splitlines() if x.strip()]
            if hits:
                hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return str(hits[0].resolve())

    fb = base_dir / "log" / "eval" / "extend_38912" / "base" / eval_tag / f"{dataset}_extend.out"
    fb.parent.mkdir(parents=True, exist_ok=True)
    return str(fb)


def scan_jobs(
    base_dir: Path,
    tags: Sequence[str],
    target: int,
    slack: int,
) -> List[Dict[str, Any]]:
    eval_outputs = base_dir / "eval_outputs"
    jobs: List[Dict[str, Any]] = []
    for tag in tags:
        d = eval_outputs / tag
        if not d.is_dir():
            print(f"[base-manifest] skip missing tag dir: {tag}")
            continue
        if "smoke" in tag:
            continue
        for metrics_path in sorted(d.glob("*think*.metrics.json")):
            if "nothink" in metrics_path.name.lower() or "smoke" in metrics_path.name.lower():
                continue
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            old_max = int(metrics.get("max_new_tokens") or 0)
            if old_max <= 0:
                continue
            already = int(metrics.get("extended_to") or 0) >= target
            if already:
                old_max = int(metrics.get("pre_extend_max_new_tokens") or old_max)
            if old_max >= target and not already:
                continue
            results_path = _results_path(metrics_path)
            if not results_path.is_file():
                continue
            try:
                results = json.loads(results_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            trunc = _count_truncations(results, old_max, target, slack)
            if trunc <= 0:
                continue
            raw_model = metrics.get("model_path") or ""
            raw_vllm = metrics.get("vllm_base_model_path") or raw_model
            model_path = _resolve_model_path(str(raw_model), base_dir) or _resolve_model_path(
                str(raw_vllm), base_dir
            )
            vllm_path = _resolve_model_path(str(raw_vllm), base_dir) or model_path
            if not model_path or not vllm_path:
                print(f"[base-manifest] unresolved model for {tag}/{metrics_path.name}: {raw_model}")
                continue
            dataset = _infer_dataset(metrics, metrics_path)
            log_out = _log_out_for(base_dir, tag, dataset, metrics_path)
            jobs.append(
                {
                    "eval_tag": tag,
                    "dataset": dataset,
                    "metrics_json": str(metrics_path.resolve()),
                    "results_json": str(results_path.resolve()),
                    "model_path": model_path,
                    "vllm_base_model_path": vllm_path,
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
                    "trunc_slack": slack,
                    "allow_long_max_model_len": True,
                    "pass_at_k_list": metrics.get("pass_at_k_list") or [1, 4, 8],
                    "relaxed_answer_extraction": bool(metrics.get("relaxed_answer_extraction", False)),
                    "log_out": log_out,
                }
            )
    return jobs


def pack_one_model_shards(jobs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for j in jobs:
        by_model.setdefault(j["model_path"], []).append(j)
    shards: List[List[Dict[str, Any]]] = []
    for _mp, group in sorted(by_model.items(), key=lambda kv: -sum(x["trunc_count"] for x in kv[1])):
        shards.append(sorted(group, key=lambda x: (x["eval_tag"], x["dataset"])))
    return shards


def batch_size_for(model_path: str) -> int:
    p = model_path.lower().replace("-", "_").replace(".", "p")
    if "8b" in p:
        return 4
    if "4b" in p:
        return 4
    if "0p6b" in p or "06b" in p:
        return 8
    return 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3]),
    )
    parser.add_argument("--target-max-new", type=int, default=TARGET_MAX_NEW)
    parser.add_argument("--trunc-slack", type=int, default=TRUNC_SLACK)
    parser.add_argument("--tags", type=str, default="", help="comma-separated override tags")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = Path(args.base_dir).expanduser().resolve()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] or list(DEFAULT_TAGS)
    out_dir = base / "scripts" / "eval" / "extend_38912" / "manifest_base"
    shard_dir = out_dir / "shards"

    jobs = scan_jobs(base, tags, args.target_max_new, args.trunc_slack)
    shards = pack_one_model_shards(jobs)
    print(
        f"[base-manifest] jobs={len(jobs)} trunc_gens={sum(j['trunc_count'] for j in jobs)} "
        f"tags={len({j['eval_tag'] for j in jobs})} shards={len(shards)}"
    )
    for i, sh in enumerate(shards):
        tag = sh[0]["eval_tag"]
        print(
            f"  shard_{i:02d}: tag={tag} jobs={len(sh)} trunc={sum(x['trunc_count'] for x in sh)} "
            f"bs={batch_size_for(sh[0]['model_path'])}"
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
    print(f"[base-manifest] wrote {man_path}")

    submit_list = out_dir / "submit_list.txt"
    with submit_list.open("w", encoding="utf-8") as sf:
        for i, sh in enumerate(shards):
            sp = shard_dir / f"shard_{i:02d}.jsonl"
            with sp.open("w", encoding="utf-8") as f:
                for j in sh:
                    f.write(json.dumps(j, ensure_ascii=False) + "\n")
            tag = sh[0]["eval_tag"]
            bs = batch_size_for(sh[0]["model_path"])
            name = f"base_{i:02d}_{tag}"[:50]
            sf.write(f"{sp}\t{bs}\t{name}\n")
            print(f"[base-manifest] wrote {sp}")

    summary = {
        "jobs": len(jobs),
        "trunc_gens": sum(j["trunc_count"] for j in jobs),
        "dirs": sorted({j["eval_tag"] for j in jobs}),
        "n_shards": len(shards),
        "target_max_new": args.target_max_new,
        "trunc_slack": args.trunc_slack,
        "allow_long_max_model_len": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[base-manifest] submit_list={submit_list}")


if __name__ == "__main__":
    main()
