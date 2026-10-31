#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HumanEval+ / MBPP+ eval via SGLang (Olmo-aligned) + official EvalPlus judging.

Prompting / judging match eval_evalplus_vllm_local.py; generation uses sglang.Engine.
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

from code_extract import infer_extract_profile
from evalplus_opsd import (
    append_jsonl_rows,
    attach_output_length_metrics,
    build_prompt,
    compute_math_style_pass_at_k,
    compute_pass_at_k_from_eval_results,
    configure_evalplus_data,
    dump_generations_json,
    load_problems,
    normalize_dataset,
    repo_root,
    run_official_evaluate,
    stop_token_ids,
    write_subset_override_jsonl,
)


def _normalize_generate_outputs(raw: Any, n_prompts: int, gen_n: int) -> List[List[str]]:
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
    p = argparse.ArgumentParser(
        description="OPSD EvalPlus (HumanEval+ / MBPP+) via SGLang + official judge"
    )
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data-jsonl", type=str, default=None)
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
    p.add_argument("--parallel", type=int, default=12)
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--force-base-prompt", action="store_true")
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
    p.add_argument("--disable-hybrid-swa-memory", action="store_true", default=True)
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    enable_thinking = bool(args.enable_thinking)
    dataset = normalize_dataset(args.dataset)

    pass_at_k_list = [int(x) for x in args.pass_at_k.split(",") if x.strip()]
    gen_n = int(args.val_n)
    pass_at_k_list = [k for k in pass_at_k_list if k <= gen_n]
    if not pass_at_k_list:
        raise SystemExit(f"pass-at-k must include a k <= val-n={gen_n}")

    data_path = configure_evalplus_data(dataset, args.data_jsonl)
    problems_map = load_problems(dataset)
    task_ids = sorted(
        problems_map.keys(), key=lambda x: (x.split("/")[0], int(x.split("/")[1]))
    )
    problems: List[Dict[str, Any]] = [problems_map[tid] for tid in task_ids]
    if int(args.max_problems) > 0:
        problems = problems[: int(args.max_problems)]
    elif args.debug:
        problems = problems[:15]
        print(f"[evalplus] debug mode: {len(problems)} problems")

    if len(problems) < len(problems_map):
        subset_path = Path(args.output_json).with_suffix(".subset.jsonl")
        data_path = write_subset_override_jsonl(dataset, problems, subset_path)
        print(f"[evalplus] subset override → {data_path} ({len(problems)} tasks)")

    print(f"[evalplus] backend=sglang attention={args.attention_backend}")
    print(f"[evalplus] dataset={dataset} data={data_path} n_problems={len(problems)}")
    print(f"[evalplus] model={args.model_path}")
    print(
        f"[evalplus] thinking={enable_thinking} n={gen_n} temp={args.temperature} "
        f"top_p={args.top_p} top_k={args.top_k} max_new_tokens={args.max_new_tokens}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    is_direct = bool(args.force_base_prompt) or getattr(tokenizer, "chat_template", None) is None
    prompts = [
        build_prompt(
            tokenizer,
            task,
            enable_thinking=enable_thinking,
            force_base_prompt=is_direct,
        )
        for task in problems
    ]
    stop_ids = stop_token_ids(tokenizer)

    # SGLang rejects when (prompt_len + max_new_tokens) >= context_length.
    # Official MiMo asks max_tokens=32768 with the same 32k model context; clamp.
    max_new_tokens = int(args.max_new_tokens)
    context_length = int(args.context_length)
    prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    max_prompt_tokens = max(prompt_lens) if prompt_lens else 0
    max_allowed_new = max(1, context_length - max_prompt_tokens - 1)
    if max_new_tokens > max_allowed_new:
        print(
            f"[evalplus] clamp max_new_tokens {max_new_tokens} -> {max_allowed_new} "
            f"(max_prompt={max_prompt_tokens} + gen must be < context_length={context_length})",
            flush=True,
        )
        max_new_tokens = max_allowed_new
    else:
        print(
            f"[evalplus] context ok: max_prompt={max_prompt_tokens} + "
            f"max_new_tokens={max_new_tokens} < context_length={context_length}",
            flush=True,
        )

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

    print("[evalplus] launching SGLang Engine ...", flush=True)
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
    if not is_direct and not enable_thinking:
        sampling_params["stop"] = ["\n```\n"]

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generations_path = out_path.with_suffix(".generations.json")
    generations_jsonl = out_path.with_suffix(".generations.jsonl")
    samples_path = out_path.with_suffix(".samples.jsonl")
    for p in (generations_jsonl, samples_path):
        if p.exists():
            p.unlink()

    save_rows: List[Dict[str, Any]] = []
    gbs = max(1, int(args.generate_batch_size))
    for start in range(0, len(prompts), gbs):
        batch_prompts = prompts[start : start + gbs]
        batch_problems = problems[start : start + len(batch_prompts)]
        print(
            f"[evalplus] generate {start}:{start + len(batch_prompts)} / {len(prompts)}",
            flush=True,
        )
        raw = llm.generate(batch_prompts, sampling_params=sampling_params)
        batch_outputs = _normalize_generate_outputs(raw, len(batch_prompts), gen_n)
        batch_rows = []
        for task, gens in zip(batch_problems, batch_outputs):
            tok_counts = [
                len(tokenizer.encode(g, add_special_tokens=False)) if g else 0
                for g in gens
            ]
            batch_rows.append(
                {
                    "task_id": task["task_id"],
                    "entry_point": task["entry_point"],
                    "output_list": gens,
                    "num_tokens_list": tok_counts,
                }
            )
        append_jsonl_rows(generations_jsonl, batch_rows)
        save_rows.extend(batch_rows)
        print(
            f"[evalplus] checkpoint {len(save_rows)}/{len(problems)} → {generations_jsonl}",
            flush=True,
        )

    dump_generations_json(generations_path, save_rows)
    print(f"[evalplus] wrote generations → {generations_path}", flush=True)

    tag = "humanevalplus" if dataset == "humaneval" else "mbppplus"
    metrics_doc: Dict[str, Any] = {
        "dataset": tag,
        "evalplus_dataset": dataset,
        "data_jsonl": str(data_path),
        "model_path": args.model_path,
        "num_problems": len(problems),
        "val_n": gen_n,
        "pass_at_k_list": pass_at_k_list,
        "enable_thinking": enable_thinking,
        "is_direct_completion": is_direct,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "max_new_tokens": max_new_tokens,
        "seed": args.seed,
        "base_only": bool(args.base_only),
        "backend": "sglang",
        "generations_json": str(generations_path),
        "extract_on_cpu": True,
        "judge": "evalplus.evaluate (official)",
    }
    len_summary = attach_output_length_metrics(metrics_doc, save_rows, tokenizer=tokenizer)
    print(
        f"[evalplus] avg_output_tokens={len_summary['avg_tokens']:.1f} "
        f"(n={len_summary['n_samples']} median={len_summary['median_tokens']:.1f} "
        f"min={len_summary['min_tokens']} max={len_summary['max_tokens']})",
        flush=True,
    )

    if not args.skip_evaluate:
        from evalplus_opsd import sanitize_generations, write_samples_jsonl

        extract_profile = infer_extract_profile(
            model_path=args.model_path,
            enable_thinking=enable_thinking,
            hint_path=str(out_path),
        )
        all_outputs = [row["output_list"] for row in save_rows]
        _raw, san_sols = sanitize_generations(
            problems,
            all_outputs,
            is_direct_completion=is_direct,
            extract_profile=extract_profile,
        )
        write_samples_jsonl(samples_path, problems, san_sols)
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
        results_path = out_path.with_suffix(".results.json")
        results_path.write_text(
            json.dumps(
                {
                    "pass_at_k_math_style": metrics_doc["pass_at_k"],
                    "pass_at_k_evalplus_unbiased": metrics_doc["evalplus_pass_at_k"],
                    "problems": math_style["problems"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        metrics_doc["results_json"] = str(results_path)
        print(
            "[evalplus] pass@k unbiased (official estimator):",
            json.dumps(metrics_doc["evalplus_pass_at_k"], indent=2),
        )
        print(
            "[evalplus] pass@k math-style any-of-k:",
            json.dumps(metrics_doc["pass_at_k"], indent=2),
        )

    out_path.write_text(json.dumps(metrics_doc, ensure_ascii=False, indent=2))
    print(f"[evalplus] wrote metrics → {out_path}")


if __name__ == "__main__":
    main()
