#!/usr/bin/env python3
"""Merge base + index-slice LCB generation shards into one ordered file for judge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row.get("question_id", ""))
        if not qid:
            raise SystemExit(f"missing question_id in {path}")
        outs = row.get("output_list") or []
        if not outs:
            raise SystemExit(f"{path} qid={qid}: empty output_list")
        out[qid] = row
    return out


def _load_problem_ids(data_jsonl: Path) -> List[str]:
    """Match eval_lcb_vllm_local.load_problems_from_jsonl order (sorted by question_id)."""
    ids: List[Any] = []
    for line in data_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = row.get("question_id")
        if qid is None:
            qid = row.get("id")
        if qid is None:
            raise SystemExit(f"cannot find question_id in {data_jsonl}")
        ids.append(qid)
    ids = sorted(ids, key=lambda x: x)
    return [str(x) for x in ids]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-jsonl", required=True)
    p.add_argument(
        "--base-jsonl",
        default="",
        help="Existing generations already completed (e.g. indices before shards).",
    )
    p.add_argument(
        "--shard-jsonl",
        action="append",
        default=[],
        help="Path to one shard .generations.jsonl (repeatable).",
    )
    p.add_argument("--output-json", required=True)
    p.add_argument("--val-n", type=int, default=4)
    args = p.parse_args()

    data_path = Path(args.data_jsonl)
    problem_ids = _load_problem_ids(data_path)
    by_id: Dict[str, Dict[str, Any]] = {}
    if args.base_jsonl:
        by_id.update(_load_jsonl_rows(Path(args.base_jsonl)))
        print(f"[merge] base={args.base_jsonl} loaded={len(by_id)}")
    for sp in args.shard_jsonl:
        part = _load_jsonl_rows(Path(sp))
        overlap = set(by_id) & set(part)
        if overlap:
            print(f"[merge] warn: {len(overlap)} qids overwritten by {sp}")
        by_id.update(part)
        print(f"[merge] shard={sp} rows={len(part)} total={len(by_id)}")

    missing = [qid for qid in problem_ids if qid not in by_id]
    if missing:
        raise SystemExit(f"missing generations for {len(missing)} problems, e.g. {missing[:5]}")

    gen_n = int(args.val_n)
    merged: List[Dict[str, Any]] = []
    for qid in problem_ids:
        row = by_id[qid]
        outs = row.get("output_list") or []
        if len(outs) < gen_n:
            raise SystemExit(f"qid={qid}: need val_n={gen_n}, got {len(outs)}")
        merged.append(row)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gens_path = out_path.with_suffix(".generations.json")
    gens_jsonl = out_path.with_suffix(".generations.jsonl")
    gens_path.write_text(json.dumps(merged, ensure_ascii=False) + "\n", encoding="utf-8")
    with gens_jsonl.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "dataset": "lcb_v6",
        "backend": "sglang",
        "merged_index_shards": True,
        "num_problems": len(merged),
        "val_n": gen_n,
        "generations_json": str(gens_path),
        "base_jsonl": args.base_jsonl or None,
        "shard_jsonls": list(args.shard_jsonl),
    }
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[merge] wrote {len(merged)} → {gens_path}")
    print(f"[merge] wrote metrics → {out_path}")


if __name__ == "__main__":
    main()
