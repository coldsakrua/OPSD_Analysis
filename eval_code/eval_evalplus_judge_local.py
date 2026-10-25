#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only EvalPlus judge: grade generations.json → metrics/results.

No GPU / vLLM / SGLang. Run on a CPU Slurm partition after generation
(see scripts/eval_code/*/*_judge.sh and evalplus_auto_submit_judge.sh).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_EVAL_CODE_DIR = Path(__file__).resolve().parent
if str(_EVAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_CODE_DIR))

from code_extract import infer_extract_profile
from evalplus_opsd import (
    attach_output_length_metrics,
    compute_avg_at_k_from_eval_results,
    compute_math_style_pass_at_k,
    compute_pass_at_k_from_eval_results,
    configure_evalplus_data,
    load_generation_rows,
    load_problems,
    normalize_dataset,
    run_official_evaluate,
    sanitize_generations,
    write_samples_jsonl,
    write_subset_override_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OPSD EvalPlus CPU judge (from generations.json)"
    )
    p.add_argument("--generations-json", type=str, required=True)
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="humaneval / humanevalplus / mbpp / mbppplus",
    )
    p.add_argument(
        "--data-jsonl",
        type=str,
        default=None,
        help="Official HumanEvalPlus.jsonl / MbppPlus.jsonl",
    )
    p.add_argument("--output-json", type=str, required=True, help="Metrics JSON path")
    p.add_argument("--pass-at-k", type=str, default="1,4")
    p.add_argument(
        "--parallel",
        type=int,
        default=48,
        help="EvalPlus worker processes (aligned with LCB NUM_PROCESS=48)",
    )
    p.add_argument(
        "--num-process-evaluate",
        dest="parallel",
        type=int,
        help="Alias of --parallel (LCB naming)",
    )
    p.add_argument("--base-only", action="store_true")
    p.add_argument(
        "--extract-profile",
        type=str,
        default="auto",
        help=(
            "strict | olmo_instruct | sft_think | auto. "
            "auto → instruct|strict only; sft_think must be set explicitly (SFT jobs)"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gens_path = Path(args.generations_json)
    out_path = Path(args.output_json)
    if not gens_path.is_file():
        raise SystemExit(f"Missing generations: {gens_path}")

    dataset = normalize_dataset(args.dataset)
    data_path = configure_evalplus_data(dataset, args.data_jsonl)
    problems_map = load_problems(dataset)

    pass_at_k_list = [int(x) for x in args.pass_at_k.split(",") if x.strip()]
    if not pass_at_k_list:
        raise SystemExit("empty --pass-at-k")

    metrics_doc: Dict[str, Any] = {}
    if out_path.is_file():
        try:
            metrics_doc = json.loads(out_path.read_text())
        except Exception:
            metrics_doc = {}
    is_direct = bool(metrics_doc.get("is_direct_completion", False))
    extract_profile = args.extract_profile
    if extract_profile in ("", "auto"):
        extract_profile = infer_extract_profile(
            model_path=str(metrics_doc.get("model_path") or ""),
            enable_thinking=metrics_doc.get("enable_thinking"),
            hint_path=str(gens_path),
        )
    print(f"[evalplus-judge] extract_profile={extract_profile}", flush=True)

    rows: List[Dict[str, Any]] = load_generation_rows(gens_path)
    ordered_problems: List[Dict[str, Any]] = []
    san_sols: List[List[str]] = []
    need_extract = 0
    for row in rows:
        tid = row["task_id"]
        if tid not in problems_map:
            raise SystemExit(f"task_id={tid} not in {data_path}")
        problem = problems_map[tid]
        gens = row.get("output_list") or []
        codes = row.get("code_list") or []
        if gens:
            _raw, san = sanitize_generations(
                [problem],
                [gens],
                is_direct_completion=is_direct,
                extract_profile=extract_profile,
            )
            codes = san[0]
            need_extract += 1
        elif not codes:
            raise SystemExit(f"empty output_list/code_list for {tid}")
        ordered_problems.append(problem)
        san_sols.append([c if isinstance(c, str) else "" for c in codes])

    gen_n = max((len(c) for c in san_sols), default=0)
    pass_at_k_list = [k for k in pass_at_k_list if k <= gen_n]
    if not pass_at_k_list:
        raise SystemExit(f"pass-at-k must include a k <= n_samples={gen_n}")

    print(
        f"[evalplus-judge] dataset={dataset} problems={len(ordered_problems)} "
        f"n_samples={gen_n} pass@k={pass_at_k_list} workers={args.parallel} "
        f"extracted_on_cpu={need_extract}",
        flush=True,
    )

    # Official evaluate requires samples cover *all* loaded problems.
    if len(ordered_problems) < len(problems_map):
        subset_path = out_path.with_suffix(".subset.jsonl")
        data_path = write_subset_override_jsonl(dataset, ordered_problems, subset_path)
        print(f"[evalplus-judge] subset override → {data_path}", flush=True)

    samples_path = out_path.with_suffix(".samples.jsonl")
    write_samples_jsonl(samples_path, ordered_problems, san_sols)
    print(f"[evalplus-judge] wrote samples → {samples_path}", flush=True)

    tag = "humanevalplus" if dataset == "humaneval" else "mbppplus"
    metrics_doc.update(
        {
            "dataset": tag,
            "evalplus_dataset": dataset,
            "data_jsonl": str(data_path),
            "num_problems": len(ordered_problems),
            "val_n": gen_n,
            "pass_at_k_list": pass_at_k_list,
            "base_only": bool(args.base_only),
            "generations_json": str(gens_path),
            "samples_jsonl": str(samples_path),
            "judge": {
                "backend": "evalplus.evaluate (official)",
                "parallel": int(args.parallel),
                "extract_profile": extract_profile,
            },
        }
    )

    result_path = run_official_evaluate(
        dataset,
        samples_path,
        parallel=int(args.parallel),
        base_only=bool(args.base_only),
        i_just_wanna_run=True,
    )
    eval_results = json.loads(result_path.read_text())
    metrics_doc["evalplus_eval_results_json"] = str(result_path)
    metrics_doc["evalplus_pass_at_k"] = compute_pass_at_k_from_eval_results(
        eval_results, pass_at_k_list, base_only=bool(args.base_only)
    )
    math_style = compute_math_style_pass_at_k(
        eval_results, pass_at_k_list, base_only=bool(args.base_only)
    )
    metrics_doc["pass_at_k"] = {
        "base": math_style["base"],
        "plus": math_style["plus"],
    }
    avg_at_k = compute_avg_at_k_from_eval_results(
        eval_results, pass_at_k_list, base_only=bool(args.base_only)
    )
    metrics_doc["avg_at_k"] = avg_at_k
    # Convenience aliases aligned with math eval (prefer plus when present).
    plus_avg = avg_at_k.get("plus") or {}
    base_avg = avg_at_k.get("base") or {}
    primary_avg = plus_avg if plus_avg else base_avg
    metrics_doc["avg1_pct"] = float(primary_avg.get("1", {}).get("rate_pct", 0.0))
    metrics_doc[f"avg{gen_n}_pct"] = float(
        primary_avg.get(str(gen_n), {}).get("rate_pct", 0.0)
    )
    if "4" in primary_avg:
        metrics_doc["avg4_pct"] = float(primary_avg["4"]["rate_pct"])

    tokenizer = None
    model_path = str(metrics_doc.get("model_path") or "")
    has_stored_lens = any(
        isinstance(r.get("num_tokens_list"), list) and r.get("num_tokens_list")
        for r in rows
    )
    if not has_stored_lens and model_path:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        except Exception as e:
            print(
                f"[evalplus-judge] warn: cannot load tokenizer for output length: {e}",
                flush=True,
            )
    len_summary = attach_output_length_metrics(
        metrics_doc, rows, tokenizer=tokenizer
    )
    print(
        f"[evalplus-judge] avg_output_tokens={len_summary['avg_tokens']:.1f} "
        f"(n={len_summary['n_samples']} median={len_summary['median_tokens']:.1f} "
        f"min={len_summary['min_tokens']} max={len_summary['max_tokens']})",
        flush=True,
    )

    results_path = out_path.with_suffix(".results.json")
    results_path.write_text(
        json.dumps(
            {
                "pass_at_k_math_style": metrics_doc["pass_at_k"],
                "pass_at_k_evalplus_unbiased": metrics_doc["evalplus_pass_at_k"],
                "avg_at_k": avg_at_k,
                "avg_output_tokens_mean": metrics_doc.get("avg_output_tokens_mean"),
                "output_length": metrics_doc.get("output_length"),
                "problems": math_style["problems"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    metrics_doc["results_json"] = str(results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics_doc, ensure_ascii=False, indent=2))

    print(
        "[evalplus-judge] pass@k unbiased:",
        json.dumps(metrics_doc["evalplus_pass_at_k"], indent=2),
    )
    print(
        "[evalplus-judge] pass@k math-style:",
        json.dumps(metrics_doc["pass_at_k"], indent=2),
    )
    print(
        "[evalplus-judge] avg@k (mean correctness):",
        json.dumps(avg_at_k, indent=2),
    )
    print(f"[evalplus-judge] wrote metrics → {out_path}")
    print(f"[evalplus-judge] wrote results → {results_path}")


if __name__ == "__main__":
    main()
