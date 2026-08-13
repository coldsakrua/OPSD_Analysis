#!/usr/bin/env python3
"""Rebuild eval_outputs/_unified_summary from *.metrics.json.

Primary metric: pass@1 (avg1_pct). Secondary: average_correct_pct.
Excludes partial_only or n!=30 from complete_n30 / conclusion tables.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval_outputs"
SUMMARY = OUT / "_unified_summary"


def parse_file(path: Path, run: str) -> dict:
    m = json.loads(path.read_text())
    name = path.name.lower()
    ds = next((d for d in ("aime24", "aime25", "aime26", "hmmt25") if d in name), None)
    # Check nothink before think: filenames end with think.metrics.json either way.
    if "nothink" in name:
        mode = "nothink"
    elif "think" in name:
        mode = "think"
    else:
        mode = "think" if m.get("enable_thinking") else "nothink"
    n = m.get("num_problems")
    partial = bool(m.get("partial_only"))
    complete = n == 30 and not partial and ds in ("aime24", "aime25", "aime26", "hmmt25")
    pak = m.get("pass_at_k") or {}
    return {
        "run": run,
        "dataset": ds,
        "mode": mode,
        "pass1": None if m.get("avg1_pct") is None else round(float(m["avg1_pct"]), 2),
        "pass4": round(pak["4"]["pct"], 2) if pak.get("4") else None,
        "pass8": round(pak["8"]["pct"], 2) if pak.get("8") else None,
        "avg_correct": None
        if m.get("average_correct_pct") is None
        else round(float(m["average_correct_pct"]), 2),
        "majority": None if m.get("majority_vote_pct") is None else round(float(m["majority_vote_pct"]), 2),
        "n": n,
        "partial": partial,
        "complete": complete,
        "file": str(path.relative_to(ROOT)),
    }


KEY_RUNS = [
    ("qwen35_4b", "Qwen3.5-4B base", "Qwen3.5"),
    ("snt_tt_1e6_ot_q35_4b_ckpt100", "snt_tt FT (OT)", "Qwen3.5"),
    ("snt_tnt_1e6_ot_q35_4b_ckpt100", "snt_tnt FT (OT)", "Qwen3.5"),
    ("st_tt_1e6_ot_q35_4b_ckpt100", "st_tt FT (OT)", "Qwen3.5"),
    ("st_tnt_1e6_ot_q35_4b_ckpt100", "st_tnt FT (OT)", "Qwen3.5"),
    ("snt_tnt_otas_q35_ckpt100", "snt_tnt otas", "Qwen3.5"),
    ("snt_tnt_ios_q35_ckpt100", "snt_tnt ios", "Qwen3.5"),
    ("snt_tnt_lora_clip005_lr5e6_q35_4b_ckpt100", "snt_tnt LoRA", "Qwen3.5"),
    ("snt_tt_lora_clip005_lr5e6_q35_4b_ckpt100", "snt_tt LoRA", "Qwen3.5"),
    ("st_tt_lora_clip005_lr5e6_q35_4b_ckpt100", "st_tt LoRA", "Qwen3.5"),
    ("qwen3-4b", "Qwen3-4B base", "Qwen3-4B"),
    ("snt_tt_1e6_ot_ckpt100", "snt_tt FT (OT)", "Qwen3-4B"),
    ("snt_tnt_1e6_ot_ckpt100", "snt_tnt FT (OT)", "Qwen3-4B"),
    ("st_tnt_1e6_ot_ckpt100", "st_tnt FT (OT)", "Qwen3-4B"),
    ("snt_tt_lora_clip005_lr5e6_4b_ckpt100", "snt_tt LoRA", "Qwen3-4B"),
    ("snt_tnt_lora_clip005_lr5e6_4b_ckpt100", "snt_tnt LoRA", "Qwen3-4B"),
    ("st_tt_lora_clip005_lr5e6_4b_ckpt100", "st_tt LoRA", "Qwen3-4B"),
    ("qwen3-4b-instruct", "4B-Instruct base", "Instruct"),
    ("snt_tt_lora_clip005_lr5e6_instruct_ckpt100", "snt_tt LoRA", "Instruct"),
    ("olmo3-7b-it", "OLMo3-7B-IT base", "OLMo"),
    ("snt_tnt_1e6_ot_olmo7bi_ckpt100", "snt_tnt FT", "OLMo"),
    ("snt_tnt_lora_clip005_lr5e6_olmo7bit_ckpt100", "snt_tnt LoRA", "OLMo"),
    ("qwen3-1.7b", "Qwen3-1.7B base", "1.7B"),
    ("snt_tt_1e6_ot_1p7b_ckpt100", "snt_tt FT", "1.7B"),
    ("snt_tnt_1e6_ot_1p7b_ckpt100", "snt_tnt FT", "1.7B"),
    ("snt_tt_lora_clip005_lr5e6_1p7b_ckpt100", "snt_tt LoRA", "1.7B"),
    ("st_tt_lora_clip005_lr5e6_1p7b_ckpt100", "st_tt LoRA", "1.7B"),
]

BASES = {
    "Qwen3.5": "qwen35_4b",
    "Qwen3-4B": "qwen3-4b",
    "Instruct": "qwen3-4b-instruct",
    "OLMo": "olmo3-7b-it",
    "1.7B": "qwen3-1.7b",
}


def main() -> None:
    SUMMARY.mkdir(exist_ok=True)
    rows: list[dict] = []
    for run_dir in sorted(OUT.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        for p in sorted(run_dir.glob("*.metrics.json")):
            rows.append(parse_file(p, run_dir.name))

    meta = {
        "primary_metric": "pass@1 (avg1_pct from metrics.json)",
        "secondary_metric": "average_correct_pct",
        "protocol": "AIME/HMMT: 30 problems x n=8. No MATH500 in this summary.",
        "n_rows": len(rows),
        "n_complete_aime_hmmt": sum(1 for r in rows if r["complete"]),
        "n_partial": sum(1 for r in rows if r["partial"]),
    }
    (SUMMARY / "README_METRICS.json").write_text(json.dumps(meta, indent=2) + "\n")

    fields = list(rows[0].keys()) if rows else []
    with (SUMMARY / "all_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    complete = [r for r in rows if r["complete"]]
    with (SUMMARY / "complete_n30.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(complete)

    by: dict[str, dict[tuple, dict]] = {}
    for r in complete:
        by.setdefault(r["run"], {})[(r["dataset"], r["mode"])] = r

    partial_by: dict[str, dict[tuple, dict]] = {}
    for r in rows:
        if r["partial"] or r["n"] not in (None, 30):
            partial_by.setdefault(r["run"], {})[(r["dataset"], r["mode"])] = r

    def cell(run: str, ds: str, mode: str, metric: str = "pass1"):
        x = by.get(run, {}).get((ds, mode))
        return x[metric] if x else None

    canvas = []
    for run, label, family in KEY_RUNS:
        row = {"run": run, "label": label, "family": family}
        nt_vals = []
        for ds in ("aime24", "aime25", "aime26", "hmmt25"):
            for mode, prefix in (("nothink", "nt"), ("think", "th")):
                row[f"{ds}_{prefix}_pass1"] = cell(run, ds, mode, "pass1")
                row[f"{ds}_{prefix}_avg"] = cell(run, ds, mode, "avg_correct")
            p = cell(run, ds, "nothink", "pass1")
            if p is not None:
                nt_vals.append(p)
        row["nt_mean_pass1"] = round(sum(nt_vals) / len(nt_vals), 2) if len(nt_vals) == 4 else None
        row["n_nt_sets"] = len(nt_vals)
        canvas.append(row)

    base_means = {
        fam: next(r["nt_mean_pass1"] for r in canvas if r["run"] == br) for fam, br in BASES.items()
    }
    for r in canvas:
        bm = base_means.get(r["family"])
        if r["nt_mean_pass1"] is not None and bm is not None and r["run"] != BASES[r["family"]]:
            r["delta_nt_mean"] = round(r["nt_mean_pass1"] - bm, 2)
        else:
            r["delta_nt_mean"] = None

    (SUMMARY / "canvas_key_runs.json").write_text(json.dumps(canvas, indent=2) + "\n")

    lines = [
        "# OPSD unified conclusions (pass@1 primary)",
        "",
        "Primary metric: **pass@1** (`avg1_pct`). Secondary: `average_correct_pct`.",
        "Only complete AIME/HMMT runs (n=30, non-partial) are included. MATH500 excluded by request.",
        "",
        "## Nothink pass@1 mean over AIME24/25/26 + HMMT25",
        "",
        "| Family | Run | NT mean pass@1 | Δ vs base |",
        "|---|---|---:|---:|",
    ]

    def fmt(v):
        return "—" if v is None else f"{v:.2f}"

    for r in canvas:
        d = "—" if r["delta_nt_mean"] is None else f"{r['delta_nt_mean']:+.2f}"
        lines.append(f"| {r['family']} | {r['label']} | {fmt(r['nt_mean_pass1'])} | {d} |")

    lines += [
        "",
        "## Per-dataset pass@1 (nothink)",
        "",
        "| Family | Run | A24 | A25 | A26 | H25 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in canvas:
        lines.append(
            f"| {r['family']} | {r['label']} | {fmt(r['aime24_nt_pass1'])} | "
            f"{fmt(r['aime25_nt_pass1'])} | {fmt(r['aime26_nt_pass1'])} | {fmt(r['hmmt25_nt_pass1'])} |"
        )

    lines += ["", "## Pending / partial (excluded)", ""]
    if not partial_by:
        lines.append("- none")
    for run, d in sorted(partial_by.items()):
        bits = [f"{ds}/{mode} n={x['n']}" for (ds, mode), x in sorted(d.items())]
        lines.append(f"- `{run}`: " + ", ".join(bits))

    (SUMMARY / "CONCLUSIONS.md").write_text("\n".join(lines) + "\n")

    pending = {
        run: {f"{ds}/{mode}": x["n"] for (ds, mode), x in d.items()}
        for run, d in partial_by.items()
    }
    (SUMMARY / "partial_status.json").write_text(json.dumps(pending, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print("partial_runs", list(pending))


if __name__ == "__main__":
    main()
