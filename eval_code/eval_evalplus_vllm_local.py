#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HumanEval+ / MBPP+ eval via vLLM (OPSD sampling) + official EvalPlus judging.

Self-contained under OPSD_Analysis:
  - generation hyperparams mirror math / LCB think|nothink scripts
  - prompts follow official EvalPlus instruct prefixes
  - sanitization + unit tests use vendored ``eval_code/evalplus``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_EVAL_CODE_DIR = Path(__file__).resolve().parent
if str(_EVAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_CODE_DIR))

from transformers import AutoTokenizer

from evalplus_opsd import (
    append_jsonl_rows,
    attach_output_length_metrics,
    build_prompt,
    completion_token_count,
    compute_math_style_pass_at_k,
    compute_pass_at_k_from_eval_results,
    configure_evalplus_data,
    dump_generations_json,
    load_problems,
    normalize_dataset,
    overlay_chat_template,
    repo_root,
    run_official_evaluate,
    stop_token_ids,
    write_subset_override_jsonl,
)


def parse_args() -> argparse.Namespace:
    repo = repo_root()
    p = argparse.ArgumentParser(
        description="OPSD EvalPlus (HumanEval+ / MBPP+) via vLLM + official judge"
    )
    p.add_argument("--model-path", type=str, required=True)
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
        help="Official HumanEvalPlus.jsonl or MbppPlus.jsonl "
        f"(default under {repo}/data/...).",
    )
    p.add_argument("--output-json", type=str, required=True)
    p.add_argument("--val-n", type=int, default=4)
    p.add_argument("--pass-at-k", type=str, default="1,4")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--generate-batch-size", type=int, default=8)
    p.add_argument("--parallel", type=int, default=12, help="EvalPlus worker processes")
    p.add_argument("--base-only", action="store_true", help="Skip plus tests")
    p.add_argument("--force-base-prompt", action="store_true")
    p.add_argument(
        "--chat-template-path",
        type=str,
        default="",
        help="Overlay tokenizer.chat_template from this model dir (instruct/think).",
    )
    p.add_argument("--enable-thinking", action="store_true", default=True)
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    p.add_argument(
        "--prefill-think",
        action="store_true",
        default=True,
        help="Append <think> to the prompt when thinking is on (Qwen3/Olmo instruct).",
    )
    p.add_argument(
        "--no-prefill-think",
        dest="prefill_think",
        action="store_false",
        help="Do not append <think> (original Base eval: chat header only).",
    )
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--debug", action="store_true", help="Only first 15 problems")
    p.add_argument("--skip-evaluate", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    enable_thinking = bool(args.enable_thinking)
    dataset = normalize_dataset(args.dataset)

    if args.temperature is None:
        args.temperature = 0.6 if enable_thinking else 0.7
    if args.top_p is None:
        args.top_p = 0.95 if enable_thinking else 0.8
    if args.max_new_tokens is None:
        args.max_new_tokens = 38912 if enable_thinking else 32768
    if args.max_model_len is None:
        args.max_model_len = 40960 if enable_thinking else 32768

    pass_at_k_list = [int(x) for x in args.pass_at_k.split(",") if x.strip()]
    gen_n = int(args.val_n)
    pass_at_k_list = [k for k in pass_at_k_list if k <= gen_n]
    if not pass_at_k_list:
        raise SystemExit(f"pass-at-k must include a k <= val-n={gen_n}")

    data_path = configure_evalplus_data(dataset, args.data_jsonl)
    problems_map = load_problems(dataset)
    task_ids = sorted(problems_map.keys(), key=lambda x: (x.split("/")[0], int(x.split("/")[1])))
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

    print(f"[evalplus] dataset={dataset} data={data_path} n_problems={len(problems)}")
    print(f"[evalplus] model={args.model_path}")
    print(
        f"[evalplus] thinking={enable_thinking} n={gen_n} temp={args.temperature} "
        f"top_p={args.top_p} top_k={args.top_k} max_new_tokens={args.max_new_tokens}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    overlay_chat_template(tokenizer, args.chat_template_path or None)
    is_direct = bool(args.force_base_prompt) or getattr(tokenizer, "chat_template", None) is None
    prompts = [
        build_prompt(
            tokenizer,
            task,
            enable_thinking=enable_thinking,
            force_base_prompt=is_direct,
            prefill_think=bool(args.prefill_think),
        )
        for task in problems
    ]
    stop_ids = stop_token_ids(tokenizer)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )

    sp_kw: Dict[str, Any] = {
        "n": gen_n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "max_tokens": args.max_new_tokens,
        "seed": args.seed,
        "presence_penalty": args.presence_penalty,
    }
    if args.top_k and args.top_k > 0:
        sp_kw["top_k"] = args.top_k
    if stop_ids:
        sp_kw["stop_token_ids"] = stop_ids
    # Official EvalPlus chat stop: after prefilled ```python, halt at the closing fence.
    # Think mode must not use this — fences often appear inside <think> and would
    # truncate before </think> / the real answer. Let EOS / max_new_tokens finish.
    if not is_direct and not enable_thinking:
        sp_kw["stop"] = list(dict.fromkeys((sp_kw.get("stop") or []) + ["\n```\n"]))
    sampling_params = SamplingParams(**sp_kw)

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
        outs = llm.generate(batch_prompts, sampling_params)
        batch_rows = []
        for task, out in zip(batch_problems, outs):
            gens: List[str] = []
            tok_counts: List[int] = []
            for o in out.outputs:
                text = o.text or ""
                gens.append(text)
                tok_counts.append(
                    completion_token_count(
                        text, getattr(o, "token_ids", None), tokenizer
                    )
                )
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
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "base_only": bool(args.base_only),
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
        from code_extract import infer_extract_profile
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
