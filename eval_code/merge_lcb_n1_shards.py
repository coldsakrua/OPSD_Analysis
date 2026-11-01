#!/usr/bin/env python3
"""Merge LCB val-n=1 shard generations into a single val-n=K file for judge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"expected list in {path}, got {type(raw)}")
    out: Dict[str, Dict[str, Any]] = {}
    for row in raw:
        qid = str(row.get("question_id", ""))
        if not qid:
            raise SystemExit(f"missing question_id in {path}")
        outs = row.get("output_list") or []
        if len(outs) < 1:
            raise SystemExit(f"{path} qid={qid}: empty output_list")
        out[qid] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shard-json",
        action="append",
        required=True,
        help="Path to one n=1 generations.json (repeat in sample order)",
    )
    p.add_argument("--output-json", required=True, help="Merged metrics/output json path")
    args = p.parse_args()

    shard_paths = [Path(x) for x in args.shard_json]
    if len(shard_paths) < 1:
        raise SystemExit("need at least one --shard-json")
    for sp in shard_paths:
        if not sp.is_file():
            raise SystemExit(f"missing shard: {sp}")

    shards = [_load_rows(sp) for sp in shard_paths]
    qids = list(shards[0].keys())
    for i, sh in enumerate(shards[1:], start=1):
        if set(sh.keys()) != set(qids):
            missing = set(qids) - set(sh.keys())
            extra = set(sh.keys()) - set(qids)
            raise SystemExit(
                f"shard {i} qid mismatch vs shard0: missing={len(missing)} extra={len(extra)}"
            )

    merged: List[Dict[str, Any]] = []
    for qid in qids:
        base = dict(shards[0][qid])
        outputs: List[str] = []
        toks: List[int] = []
        for sh in shards:
            row = sh[qid]
            outputs.append((row.get("output_list") or [""])[0] or "")
            tlist = row.get("num_tokens_list") or [0]
            toks.append(int(tlist[0]) if tlist else 0)
        base["output_list"] = outputs
        base["num_tokens_list"] = toks
        merged.append(base)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gens_path = out_path.with_suffix(".generations.json")
    gens_jsonl = out_path.with_suffix(".generations.jsonl")

    gens_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with gens_jsonl.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_tok = [t for row in merged for t in (row.get("num_tokens_list") or [])]
    metrics = {
        "merged_from": [str(p) for p in shard_paths],
        "num_problems": len(merged),
        "val_n": len(shard_paths),
        "generations_json": str(gens_path),
        "avg_output_tokens_mean": (sum(n_tok) / len(n_tok)) if n_tok else 0.0,
        "output_length": {
            "n_samples": len(n_tok),
            "avg_tokens": (sum(n_tok) / len(n_tok)) if n_tok else 0.0,
            "median_tokens": sorted(n_tok)[len(n_tok) // 2] if n_tok else 0,
            "min_tokens": min(n_tok) if n_tok else 0,
            "max_tokens": max(n_tok) if n_tok else 0,
        },
    }
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[merge] problems={len(merged)} val_n={len(shard_paths)} → {gens_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
