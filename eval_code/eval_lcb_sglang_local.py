#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LiveCodeBench v6 eval via SGLang (for models without native vLLM, e.g. Olmo-3).

Prompting / judging match eval_lcb_vllm_local.py; generation uses sglang.Engine
with the same knobs as scripts/eval/olmo_*/*_sgl.sh.
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

from transformers import AutoTokenizer

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.evaluation import codegen_metrics
from lcb_runner.evaluation.pass_k_utils import extract_instance_results
from code_extract import extract_lcb_candidate, infer_extract_profile
from evalplus_opsd import attach_output_length_metrics

# Reuse prompt builders from the vLLM entrypoint.
from eval_lcb_vllm_local import (  # noqa: E402
    _apply_chat_prompt,
    _stop_token_ids,
    build_prompt,
    load_problems_from_jsonl,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_generate_outputs(raw: Any, n_prompts: int, gen_n: int) -> List[List[str]]:
    """Map Engine.generate return value to [prompt][sample] text strings."""
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise RuntimeError(f"unexpected SGLang generate return type: {type(raw)}")

    if len(raw) == n_prompts * gen_n:
        grouped: List[List[str]] = []
        for i in range(n_prompts):
            chunk = raw[i * gen_n : (i + 1) * gen_n]
            texts = []
            for item in chunk:
                t = item.get("text") if isinstance(item, dict) else item
                if isinstance(t, list):
                    t = t[0] if t else ""
                texts.append(t or "")
            grouped.append(texts)
        return grouped

    if len(raw) == n_prompts:
        grouped = []
        for item in raw:
            texts = item.get("text") if isinstance(item, dict) else item
            if isinstance(texts, list):
                if len(texts) < gen_n:
                    raise RuntimeError(f"expected {gen_n} texts, got {len(texts)}")
                grouped.append([(t or "") for t in texts[:gen_n]])
            else:
                if gen_n != 1:
                    raise RuntimeError(f"expected n={gen_n} samples, got single text")
                grouped.append([texts or ""])
        return grouped

    raise RuntimeError(
        f"cannot reshape SGLang outputs: len={len(raw)} n_prompts={n_prompts} gen_n={gen_n}"
    )


def parse_args() -> argparse.Namespace:
    repo = _repo_root()
    p = argparse.ArgumentParser(description="OPSD LCB v6 eval via SGLang (Olmo-aligned)")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument(
        "--data-jsonl",
        type=str,
        default=str(repo / "data" / "livecodebench" / "test6.jsonl"),
    )
    p.add_argument("--output-json", type=str, required=True)
    p.add_argument("--val-n", type=int, default=4)
    p.add_argument("--pass-at-k", type=str, default="1,4")
    p.add_argument("--max-new-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generate-batch-size", type=int, default=4)
    p.add_argument("--timeout", type=int, default=6)
    p.add_argument("--num-process-evaluate", type=int, default=12)
    p.add_argument("--enable-thinking", action="store_true", default=True)
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    p.add_argument("--attention-backend", type=str, default="triton")
    p.add_argument(
        "--sampling-backend",
        type=str,
        default="pytorch",
        choices=["pytorch", "flashinfer", "ascend"],
    )
    p.add_argument("--mem-fraction-static", type=float, default=0.80)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--context-length", type=int, default=40960)
    p.add_argument(
        "--disable-hybrid-swa-memory",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--reasoning-parser",
        type=str,
        default="",
        help="SGLang reasoning parser name (e.g. deepseek-r1). Empty disables.",
    )
    p.add_argument(
        "--disable-piecewise-cuda-graph",
        action="store_true",
        default=False,
        help=(
            "Pass disable_piecewise_cuda_graph to SGLang Engine. "
            "Required for Falcon-H1: piecewise CUDA graph warmup hits Dynamo KeyError."
        ),
    )
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--skip-evaluate", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Keep existing .generations.jsonl and skip completed question_ids.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    enable_thinking = bool(args.enable_thinking)

    pass_at_k_list = [int(x) for x in args.pass_at_k.split(",") if x.strip()]
    gen_n = int(args.val_n)
    pass_at_k_list = [k for k in pass_at_k_list if k <= gen_n]
    if not pass_at_k_list:
        raise SystemExit(f"pass-at-k must include a k <= val-n={gen_n}")

    data_path = Path(args.data_jsonl)
    if not data_path.is_file():
        raise SystemExit(f"Missing data: {data_path}")

    problems = load_problems_from_jsonl(data_path, max_problems=int(args.max_problems))
    if args.debug and int(args.max_problems) <= 0:
        problems = problems[:15]
        print(f"[lcb] debug mode: {len(problems)} problems")

    print(f"[lcb] backend=sglang attention={args.attention_backend}")
    print(f"[lcb] model={args.model_path}")
    print(
        f"[lcb] thinking={enable_thinking} n={gen_n} temp={args.temperature} "
        f"top_p={args.top_p} top_k={args.top_k} max_new_tokens={args.max_new_tokens}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    all_problems = list(problems)
    all_prompts = [build_prompt(tokenizer, prob, enable_thinking) for prob in all_problems]
    stop_ids = _stop_token_ids(tokenizer)
    n_total = len(all_problems)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generations_path = out_path.with_suffix(".generations.json")
    generations_jsonl = out_path.with_suffix(".generations.jsonl")

    save_rows: List[Dict[str, Any]] = []
    done_ids: set[str] = set()
    if args.resume and generations_jsonl.exists():
        for line in generations_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = str(row.get("question_id", ""))
            if not qid or qid in done_ids:
                continue
            outs = row.get("output_list") or []
            if len(outs) < gen_n:
                continue
            done_ids.add(qid)
            save_rows.append(row)
        print(
            f"[lcb] resume: loaded {len(save_rows)}; "
            f"remaining {n_total - len(done_ids)}/{n_total}",
            flush=True,
        )
    elif generations_jsonl.exists():
        generations_jsonl.unlink()

    keep_idx = [
        i
        for i, prob in enumerate(all_problems)
        if str(prob.question_id) not in done_ids
    ]
    problems = [all_problems[i] for i in keep_idx]
    prompts = [all_prompts[i] for i in keep_idx]

    # SGLang rejects when (prompt_len + max_new_tokens) >= context_length.
    # Official MiMo asks max_tokens=32768 with the same 32k model context; clamp.
    max_new_tokens = int(args.max_new_tokens)
    context_length = int(args.context_length)
    prompt_lens = [
        len(tokenizer.encode(p, add_special_tokens=False)) for p in all_prompts
    ]
    max_prompt_tokens = max(prompt_lens) if prompt_lens else 0
    max_allowed_new = max(1, context_length - max_prompt_tokens - 1)
    if max_new_tokens > max_allowed_new:
        print(
            f"[lcb] clamp max_new_tokens {max_new_tokens} -> {max_allowed_new} "
            f"(max_prompt={max_prompt_tokens} + gen must be < context_length={context_length})",
            flush=True,
        )
        max_new_tokens = max_allowed_new
    else:
        print(
            f"[lcb] context ok: max_prompt={max_prompt_tokens} + "
            f"max_new_tokens={max_new_tokens} < context_length={context_length}",
            flush=True,
        )

    llm = None
    if prompts:
        from sglang import Engine

        engine_kwargs: Dict[str, Any] = {
            "model_path": args.model_path,
            "tokenizer_path": args.model_path,
            "trust_remote_code": True,
            "attention_backend": args.attention_backend,
            "sampling_backend": args.sampling_backend,
            "mem_fraction_static": args.mem_fraction_static,
            "tp_size": args.tp_size,
            "context_length": context_length,
            "log_level": "warning",
        }
        if args.disable_hybrid_swa_memory:
            engine_kwargs["disable_hybrid_swa_memory"] = True
        if args.reasoning_parser:
            engine_kwargs["reasoning_parser"] = args.reasoning_parser
        if args.disable_piecewise_cuda_graph:
            engine_kwargs["disable_piecewise_cuda_graph"] = True

        print(f"[lcb] launching SGLang Engine ...", flush=True)
        llm = Engine(**engine_kwargs)

        sampling_params: Dict[str, Any] = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": max_new_tokens,
            "n": gen_n,
            "sampling_seed": args.seed,
            "skip_special_tokens": True,
        }
        if args.top_k and args.top_k > 0:
            sampling_params["top_k"] = args.top_k
        if args.min_p and args.min_p > 0:
            sampling_params["min_p"] = args.min_p
        if args.presence_penalty != 0.0:
            sampling_params["presence_penalty"] = args.presence_penalty
        if stop_ids:
            sampling_params["stop_token_ids"] = stop_ids

        gbs = max(1, int(args.generate_batch_size))
        n_done_before = len(save_rows)
        for start in range(0, len(prompts), gbs):
            batch_prompts = prompts[start : start + gbs]
            batch_problems = problems[start : start + len(batch_prompts)]
            abs_start = n_done_before + start
            print(
                f"[lcb] generate {abs_start}:{abs_start + len(batch_prompts)} / {n_total}",
                flush=True,
            )
            raw = llm.generate(batch_prompts, sampling_params=sampling_params)
            batch_outputs = _normalize_generate_outputs(raw, len(batch_prompts), gen_n)
            batch_rows = []
            for problem, gens in zip(batch_problems, batch_outputs):
                tok_counts = [
                    len(tokenizer.encode(g, add_special_tokens=False)) if g else 0
                    for g in gens
                ]
                batch_rows.append(
                    {
                        "question_id": problem.question_id,
                        "question_title": problem.question_title,
                        "difficulty": problem.difficulty.value,
                        "platform": problem.platform.value,
                        "contest_date": problem.contest_date.isoformat(),
                        "output_list": gens,
                        "num_tokens_list": tok_counts,
                    }
                )
            with generations_jsonl.open("a", encoding="utf-8") as f:
                for row in batch_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            save_rows.extend(batch_rows)
            print(
                f"[lcb] checkpoint {len(save_rows)}/{n_total} → {generations_jsonl}",
                flush=True,
            )
    else:
        print(f"[lcb] resume: nothing left to generate ({n_total}/{n_total})", flush=True)

    by_id = {str(r["question_id"]): r for r in save_rows}
    missing = [str(p.question_id) for p in all_problems if str(p.question_id) not in by_id]
    if missing:
        raise SystemExit(f"missing generations for {len(missing)} problems, e.g. {missing[:5]}")
    save_rows = [by_id[str(p.question_id)] for p in all_problems]
    problems = all_problems

    generations_path.write_text(json.dumps(save_rows, ensure_ascii=False))
    print(f"[lcb] wrote generations → {generations_path}", flush=True)

    metrics_doc: Dict[str, Any] = {
        "dataset": "lcb_v6",
        "backend": "sglang",
        "attention_backend": args.attention_backend,
        "data_jsonl": str(data_path),
        "model_path": args.model_path,
        "num_problems": len(problems),
        "val_n": gen_n,
        "pass_at_k_list": pass_at_k_list,
        "enable_thinking": enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "max_new_tokens": max_new_tokens,
        "seed": args.seed,
        "generate_batch_size": max(1, int(args.generate_batch_size)),
        "mem_fraction_static": args.mem_fraction_static,
        "context_length": args.context_length,
        "tp_size": args.tp_size,
        "generations_json": str(generations_path),
        "extract_on_cpu": True,
    }
    len_summary = attach_output_length_metrics(metrics_doc, save_rows, tokenizer=tokenizer)
    print(
        f"[lcb] avg_output_tokens={len_summary['avg_tokens']:.1f} "
        f"(n={len_summary['n_samples']} median={len_summary['median_tokens']:.1f} "
        f"min={len_summary['min_tokens']} max={len_summary['max_tokens']})",
        flush=True,
    )

    if not args.skip_evaluate:
        extract_profile = infer_extract_profile(
            model_path=args.model_path,
            enable_thinking=enable_thinking,
            hint_path=str(out_path),
        )
        extracted = [
            [extract_lcb_candidate(g, profile=extract_profile) for g in row["output_list"]]
            for row in save_rows
        ]
        eval_samples = [prob.get_evaluation_sample() for prob in problems]
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
        for i, problem in enumerate(problems):
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

        n = len(problems)
        metrics_doc["pass_at_k"] = {
            str(k): {
                "n_solved": pass_counts[k],
                "n_total": n,
                "rate": pass_counts[k] / n if n else 0.0,
                "rate_pct": 100.0 * pass_counts[k] / n if n else 0.0,
            }
            for k in pass_at_k_list
        }
        results_path = out_path.with_suffix(".results.json")
        results_path.write_text(
            json.dumps(
                {
                    "pass_at_k_math_style": metrics_doc["pass_at_k"],
                    "pass_at_k_lcb_unbiased": metrics_doc["lcb_pass_at_k"],
                    "problems": per_problem,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        metrics_doc["results_json"] = str(results_path)
        print("[lcb] pass@k (math-style any-of-k):", json.dumps(metrics_doc["pass_at_k"], indent=2))
        print("[lcb] pass@k (LCB unbiased estimator):", json.dumps(metrics_doc["lcb_pass_at_k"], indent=2))

    out_path.write_text(json.dumps(metrics_doc, ensure_ascii=False, indent=2))
    print(f"[lcb] wrote metrics → {out_path}")

    if llm is not None:
        try:
            llm.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
