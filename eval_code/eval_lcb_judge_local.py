#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only LiveCodeBench judge: grade generations.json → metrics/results.

No GPU / vLLM / SGLang. Intended to run on a CPU Slurm partition after
generation finishes (see scripts/eval_code/run_lcb_v6_judge.sh).
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

from evalplus_opsd import attach_output_length_metrics, load_generation_rows
from lcb_runner.evaluation import codegen_metrics
from lcb_runner.evaluation.pass_k_utils import extract_instance_results

from eval_lcb_vllm_local import load_problems_from_jsonl
from code_extract import extract_lcb_candidate, infer_extract_profile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OPSD LCB v6 CPU judge (from generations.json)")
    p.add_argument("--generations-json", type=str, required=True)
    p.add_argument("--data-jsonl", type=str, required=True)
    p.add_argument("--output-json", type=str, required=True, help="Metrics JSON path")
    p.add_argument("--pass-at-k", type=str, default="1,4")
    p.add_argument("--timeout", type=int, default=6)
    p.add_argument("--num-process-evaluate", type=int, default=16)
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
    data_path = Path(args.data_jsonl)
    out_path = Path(args.output_json)
    if not gens_path.is_file():
        raise SystemExit(f"Missing generations: {gens_path}")
    if not data_path.is_file():
        raise SystemExit(f"Missing data: {data_path}")

    pass_at_k_list = [int(x) for x in args.pass_at_k.split(",") if x.strip()]
    if not pass_at_k_list:
        raise SystemExit("empty --pass-at-k")

    metrics_doc: Dict[str, Any] = {}
    if out_path.is_file():
        try:
            metrics_doc = json.loads(out_path.read_text())
        except Exception:
            metrics_doc = {}

    extract_profile = args.extract_profile
    if extract_profile in ("", "auto"):
        extract_profile = infer_extract_profile(
            model_path=str(metrics_doc.get("model_path") or ""),
            enable_thinking=metrics_doc.get("enable_thinking"),
            hint_path=str(gens_path),
        )
    print(f"[lcb-judge] extract_profile={extract_profile}", flush=True)

    rows: List[Dict[str, Any]] = load_generation_rows(gens_path)
    problems = load_problems_from_jsonl(data_path)
    by_qid = {p.question_id: p for p in problems}

    ordered_problems = []
    extracted: List[List[str]] = []
    need_extract = 0
    for row in rows:
        qid = row["question_id"]
        if qid not in by_qid:
            raise SystemExit(f"question_id={qid} not in {data_path}")
        gens = row.get("output_list") or []
        codes = row.get("code_list") or []
        if gens:
            # Always re-extract from raw generations: stored code_list may be
            # empty from the official two-fence extractor (common for Olmo).
            codes = [
                extract_lcb_candidate(g, profile=extract_profile) for g in gens
            ]
            need_extract += 1
        elif not codes:
            raise SystemExit(f"empty output_list/code_list for {qid}")
        ordered_problems.append(by_qid[qid])
        extracted.append([c if isinstance(c, str) else "" for c in codes])

    gen_n = max((len(c) for c in extracted), default=0)
    pass_at_k_list = [k for k in pass_at_k_list if k <= gen_n]
    if not pass_at_k_list:
        raise SystemExit(f"pass-at-k must include a k <= n_samples={gen_n}")

    print(
        f"[lcb-judge] problems={len(ordered_problems)} n_samples={gen_n} "
        f"pass@k={pass_at_k_list} workers={args.num_process_evaluate} "
        f"extracted_on_cpu={need_extract}",
        flush=True,
    )

    metrics_doc.update(
        {
            "dataset": metrics_doc.get("dataset", "lcb_v6"),
            "data_jsonl": str(data_path),
            "num_problems": len(ordered_problems),
            "val_n": gen_n,
            "pass_at_k_list": pass_at_k_list,
            "generations_json": str(gens_path),
            "judge": {
                "timeout": args.timeout,
                "num_process_evaluate": args.num_process_evaluate,
                "extract": "relaxed_from_output_list",
                "extract_profile": extract_profile,
            },
        }
    )

    eval_samples = [prob.get_evaluation_sample() for prob in ordered_problems]
    metrics, results, _metadata = codegen_metrics(
        eval_samples,
        extracted,
        k_list=pass_at_k_list,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
        debug=False,
    )
    metrics_doc["lcb_pass_at_k"] = {
        k: float(v) for k, v in metrics.items() if k.startswith("pass@") and not isinstance(v, dict)
    }

    instance_grades = extract_instance_results(results)
    pass_counts = {k: 0 for k in pass_at_k_list}
    per_problem = []
    for i, problem in enumerate(ordered_problems):
        graded_list = [bool(x) for x in (instance_grades[i] if i < len(instance_grades) else [])]
        ok_any = {k: (len(graded_list) >= 1 and any(graded_list[:k])) for k in pass_at_k_list}
        for k in pass_at_k_list:
            if ok_any[k]:
                pass_counts[k] += 1
        per_problem.append(
            {
                "question_id": problem.question_id,
                "difficulty": problem.difficulty.value,
                "graded_list": graded_list,
                "pass_at_k": ok_any,
            }
        )

    n = len(ordered_problems)
    metrics_doc["pass_at_k"] = {
        str(k): {
            "n_solved": pass_counts[k],
            "n_total": n,
            "rate": pass_counts[k] / n if n else 0.0,
            "rate_pct": 100.0 * pass_counts[k] / n if n else 0.0,
        }
        for k in pass_at_k_list
    }
    # avg@k: mean correctness over first-k samples (same as math eval avg{n}_pct).
    avg_at_k: Dict[str, Any] = {}
    for k in pass_at_k_list:
        n_correct = 0
        n_total = 0
        for pp in per_problem:
            gl = pp["graded_list"][:k]
            n_correct += sum(1 for x in gl if x)
            n_total += len(gl)
        rate = (n_correct / n_total) if n_total else 0.0
        avg_at_k[str(k)] = {
            "n_correct": n_correct,
            "n_total": n_total,
            "rate": rate,
            "rate_pct": 100.0 * rate,
        }
    metrics_doc["avg_at_k"] = avg_at_k
    metrics_doc["avg1_pct"] = float(avg_at_k.get("1", {}).get("rate_pct", 0.0))
    metrics_doc[f"avg{gen_n}_pct"] = float(
        avg_at_k.get(str(gen_n), {}).get("rate_pct", 0.0)
    )
    if "4" in avg_at_k:
        metrics_doc["avg4_pct"] = float(avg_at_k["4"]["rate_pct"])

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
                f"[lcb-judge] warn: cannot load tokenizer for output length: {e}",
                flush=True,
            )
    len_summary = attach_output_length_metrics(
        metrics_doc, rows, tokenizer=tokenizer
    )
    print(
        f"[lcb-judge] avg_output_tokens={len_summary['avg_tokens']:.1f} "
        f"(n={len_summary['n_samples']} median={len_summary['median_tokens']:.1f} "
        f"min={len_summary['min_tokens']} max={len_summary['max_tokens']})",
        flush=True,
    )

    results_path = out_path.with_suffix(".results.json")
    results_path.write_text(
        json.dumps(
            {
                "pass_at_k_math_style": metrics_doc["pass_at_k"],
                "pass_at_k_lcb_unbiased": metrics_doc["lcb_pass_at_k"],
                "avg_at_k": avg_at_k,
                "avg_output_tokens_mean": metrics_doc.get("avg_output_tokens_mean"),
                "output_length": metrics_doc.get("output_length"),
                "problems": per_problem,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    metrics_doc["results_json"] = str(results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics_doc, ensure_ascii=False, indent=2))

    print("[lcb-judge] pass@k (math-style any-of-k):", json.dumps(metrics_doc["pass_at_k"], indent=2))
    print("[lcb-judge] pass@k (LCB unbiased estimator):", json.dumps(metrics_doc["lcb_pass_at_k"], indent=2))
    print("[lcb-judge] avg@k (mean correctness):", json.dumps(avg_at_k, indent=2))
    print(f"[lcb-judge] wrote metrics → {out_path}")
    print(f"[lcb-judge] wrote results → {results_path}")


if __name__ == "__main__":
    main()
