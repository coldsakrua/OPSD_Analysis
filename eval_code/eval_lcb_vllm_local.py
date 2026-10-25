#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained LiveCodeBench (v6) eval under OPSD_Analysis.

Generation hyperparams mirror math eval (think / nothink).
Judging uses the vendored official LCB checker (eval_code/lcb_runner).
No Absolute-Zero-Reasoner / SDPO path dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make vendored lcb_runner importable from this file's directory.
_EVAL_CODE_DIR = Path(__file__).resolve().parent
if str(_EVAL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_CODE_DIR))

from transformers import AutoTokenizer

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.evaluation import codegen_metrics
from lcb_runner.evaluation.pass_k_utils import extract_instance_results
from lcb_runner.prompts.code_generation import PromptConstants
from code_extract import extract_lcb_candidate, infer_extract_profile
from evalplus_opsd import attach_output_length_metrics, completion_token_count


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _apply_chat_prompt(tokenizer: Any, messages: List[Dict[str, str]], enable_thinking: bool) -> str:
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        kwargs["enable_thinking"] = bool(enable_thinking)
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _stop_token_ids(tokenizer: Any) -> List[int]:
    ids: List[int] = []
    for tok in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0 and tid not in ids:
                ids.append(tid)
        except Exception:
            pass
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int) and eos not in ids:
        ids.append(eos)
    elif isinstance(eos, list):
        for x in eos:
            if isinstance(x, int) and x not in ids:
                ids.append(x)
    return ids


def load_problems_from_jsonl(path: Path, max_problems: int = 0) -> List[CodeGenerationProblem]:
    """Read jsonl. If max_problems>0, stop after the first N records (file order, no full scan)."""
    problems: List[CodeGenerationProblem] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(CodeGenerationProblem(**json.loads(line)))
            if max_problems > 0 and len(problems) >= max_problems:
                break
    if max_problems <= 0:
        problems = sorted(problems, key=lambda x: x.question_id)
    print(f"[lcb] loaded {len(problems)} problems from {path}")
    return problems


def build_user_prompt(problem: CodeGenerationProblem) -> str:
    """Match official LCB Qwen user-turn content (code_generation.get_qwen_...)."""
    prompt = (
        "You will be given a question (problem specification) and will generate a correct "
        "Python program that matches the specification and passes all tests. "
        "You will NOT return anything except for the program.\n\n"
    )
    prompt += f"Question:\n{problem.question_content}\n\n"
    if problem.starter_code:
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    return prompt


def build_prompt(
    tokenizer: Any,
    problem: CodeGenerationProblem,
    enable_thinking: bool,
    prefill_think: bool = True,
) -> str:
    messages = [
        {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        {"role": "user", "content": build_user_prompt(problem)},
    ]
    text = _apply_chat_prompt(tokenizer, messages, enable_thinking=enable_thinking)
    if enable_thinking and prefill_think:
        from evalplus_opsd import ensure_think_open

        text = ensure_think_open(text)
    return text


def parse_args() -> argparse.Namespace:
    repo = _repo_root()
    p = argparse.ArgumentParser(description="OPSD local LiveCodeBench v6 eval (math-aligned sampling)")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument(
        "--data-jsonl",
        type=str,
        default=str(repo / "data" / "livecodebench" / "test6.jsonl"),
        help="Official LCB v6 slice jsonl (local copy under data/livecodebench).",
    )
    p.add_argument("--output-json", type=str, required=True)
    p.add_argument("--val-n", type=int, default=4, help="Samples per problem")
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
    p.add_argument("--timeout", type=int, default=6, help="Per-test timeout (LCB default 6)")
    p.add_argument("--num-process-evaluate", type=int, default=12)
    p.add_argument("--enable-thinking", action="store_true", default=True)
    p.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    p.add_argument(
        "--chat-template-path",
        type=str,
        default="",
        help="Overlay tokenizer.chat_template from this model dir (instruct/think).",
    )
    p.add_argument(
        "--prefill-think",
        action="store_true",
        default=True,
        help="Append <think> to the prompt when thinking is on.",
    )
    p.add_argument(
        "--no-prefill-think",
        dest="prefill_think",
        action="store_false",
        help="Do not append <think> (original Base eval).",
    )
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument(
        "--max-problems",
        type=int,
        default=0,
        help="If >0, keep only the first N problems (sorted by question_id).",
    )
    p.add_argument("--debug", action="store_true", help="Only first 15 problems")
    p.add_argument("--skip-evaluate", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    enable_thinking = bool(args.enable_thinking)

    # Align defaults with math eval scripts under scripts/eval/4b/
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

    data_path = Path(args.data_jsonl)
    if not data_path.is_file():
        raise SystemExit(f"Missing data: {data_path}")

    problems = load_problems_from_jsonl(data_path, max_problems=int(args.max_problems))
    if args.debug and int(args.max_problems) <= 0:
        problems = problems[:15]
        print(f"[lcb] debug mode: {len(problems)} problems")

    print(f"[lcb] model={args.model_path}")
    print(
        f"[lcb] thinking={enable_thinking} n={gen_n} temp={args.temperature} "
        f"top_p={args.top_p} top_k={args.top_k} max_new_tokens={args.max_new_tokens}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    from evalplus_opsd import overlay_chat_template

    overlay_chat_template(tokenizer, args.chat_template_path or None)
    prompts = [
        build_prompt(
            tokenizer,
            prob,
            enable_thinking,
            prefill_think=bool(args.prefill_think),
        )
        for prob in problems
    ]
    stop_ids = _stop_token_ids(tokenizer)

    # Prefer math-eval-like vLLM env (set by the shell script).
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
    sampling_params = SamplingParams(**sp_kw)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generations_path = out_path.with_suffix(".generations.json")
    generations_jsonl = out_path.with_suffix(".generations.jsonl")
    if generations_jsonl.exists():
        generations_jsonl.unlink()

    save_rows: List[Dict[str, Any]] = []
    gbs = max(1, int(args.generate_batch_size))
    for start in range(0, len(prompts), gbs):
        batch_prompts = prompts[start : start + gbs]
        batch_problems = problems[start : start + len(batch_prompts)]
        print(
            f"[lcb] generate {start}:{start + len(batch_prompts)} / {len(prompts)}",
            flush=True,
        )
        outs = llm.generate(batch_prompts, sampling_params)
        batch_rows = []
        for problem, out in zip(batch_problems, outs):
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
            f"[lcb] checkpoint {len(save_rows)}/{len(problems)} → {generations_jsonl}",
            flush=True,
        )

    generations_path.write_text(json.dumps(save_rows, ensure_ascii=False))
    print(f"[lcb] wrote generations → {generations_path}", flush=True)

    metrics_doc: Dict[str, Any] = {
        "dataset": "lcb_v6",
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
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
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
            hint_path=str(generations_path),
        )
        extracted = [
            [extract_lcb_candidate(g, profile=extract_profile) for g in row["output_list"]]
            for row in save_rows
        ]
        eval_samples = [prob.get_evaluation_sample() for prob in problems]
        # Official LCB: returns [metrics_dict, results_dict, metadata_list]
        metrics, results, _metadata = codegen_metrics(
            eval_samples,
            extracted,
            k_list=pass_at_k_list,
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.timeout,
            debug=False,
        )
        # metrics: {"pass@1": float, "pass@4": float, ..., "detail": {...}}
        metrics_doc["lcb_pass_at_k"] = {
            k: float(v) for k, v in metrics.items() if k.startswith("pass@") and not isinstance(v, dict)
        }

        # Per-problem bool grades (True if all unit tests passed for that sample)
        instance_grades = extract_instance_results(results)  # List[List[bool]]
        pass_counts = {k: 0 for k in pass_at_k_list}
        per_problem = []
        for i, problem in enumerate(problems):
            graded_list = [bool(x) for x in (instance_grades[i] if i < len(instance_grades) else [])]
            # Math-style pass@k: any of first k samples correct
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


if __name__ == "__main__":
    main()
