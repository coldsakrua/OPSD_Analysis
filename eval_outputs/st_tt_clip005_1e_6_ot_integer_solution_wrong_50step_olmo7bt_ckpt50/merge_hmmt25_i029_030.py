#!/usr/bin/env python3
"""Merge main(1-28) + i029_030(29-30) for wrong_50step ckpt50 hmmt25."""
import json
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
main_r = base / "hmmt25_olmo_7b_think_sgl_seed42.results.json"
shard_r = base / "hmmt25_olmo_7b_think_sgl_seed42.i029_030.results.json"
out_metrics = base / "hmmt25_olmo_7b_think_sgl_seed42.metrics.json"
out_results = base / "hmmt25_olmo_7b_think_sgl_seed42.results.json"
out_json = base / "hmmt25_olmo_7b_think_sgl_seed42.json"

def load_rows(p: Path):
    doc = json.loads(p.read_text())
    rows = doc["results"] if isinstance(doc, dict) else doc
    return {str(r["problem_id"]): r for r in rows}

def need(src, ids, label):
    miss = [i for i in ids if str(i) not in src]
    if miss:
        sys.exit(f"{label} missing ids {miss}")
    for i in ids:
        if src[str(i)].get("_shard_stub"):
            sys.exit(f"{label} id {i} is still a stub")

main, shard = load_rows(main_r), load_rows(shard_r)
need(main, range(1, 29), "main")
need(shard, range(29, 31), "i029_030")
merged = [main[str(i)] for i in range(1, 29)] + [shard[str(i)] for i in range(29, 31)]
assert len(merged) == 30

pass_at_k_list = [1, 4, 8]
pass_at_k_counts = {k: 0 for k in pass_at_k_list}
formatted_total = total_solutions = total_correct = total_output_tokens = majority_correct = 0
for r in merged:
    for g in r.get("generations") or []:
        total_solutions += 1
        total_output_tokens += int(g.get("num_tokens") or 0)
        if g.get("formatted"):
            formatted_total += 1
        if g.get("correct"):
            total_correct += 1
    for k in pass_at_k_list:
        if r.get("pass_at_k", {}).get(str(k)):
            pass_at_k_counts[k] += 1
    if r.get("majority_vote_correct"):
        majority_correct += 1
n = len(merged)
summary = {
    "partial_only": False,
    "results_count": n,
    "num_problems": n,
    "merged_from": [str(main_r), str(shard_r)],
    "pass_at_k": {
        str(k): {"count": pass_at_k_counts[k], "total": n, "pct": 100.0 * pass_at_k_counts[k] / n}
        for k in pass_at_k_list
    },
    "majority_vote_acc": {"count": majority_correct, "total": n, "pct": 100.0 * majority_correct / n},
    "solution_acc": {
        "correct": total_correct,
        "total": total_solutions,
        "pct": 100.0 * total_correct / total_solutions if total_solutions else 0.0,
    },
    "formatted_rate": {
        "formatted": formatted_total,
        "total": total_solutions,
        "pct": 100.0 * formatted_total / total_solutions if total_solutions else 0.0,
    },
    "avg_output_tokens": total_output_tokens / total_solutions if total_solutions else 0.0,
}
backup = out_results.with_suffix(out_results.suffix + ".pre_merge.bak")
if out_results.exists() and not backup.exists():
    backup.write_bytes(out_results.read_bytes())
out_metrics.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
out_results.write_text(
    json.dumps(
        {"partial_only": False, "results_count": n, "num_problems_total": 30, "metrics_json": str(out_metrics), "results": merged},
        ensure_ascii=False,
        separators=(",", ":"),
    ),
    encoding="utf-8",
)
out_json.write_text(json.dumps({**summary, "results_json": str(out_results)}, indent=2), encoding="utf-8")
print(f"[merge] n={n}")
for k in pass_at_k_list:
    print(f"  pass@{k}: {summary['pass_at_k'][str(k)]['pct']:.2f}%")
print(f"  majority: {summary['majority_vote_acc']['pct']:.2f}%")
