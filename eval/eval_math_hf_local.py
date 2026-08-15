#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Math eval with HuggingFace transformers.generate (for backends vLLM cannot run well, e.g. Olmo-3)."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from verl_rlsd.olmo_chat_template import (
    is_olmo_instruct_model_path,
    is_olmo_model_path,
    is_olmo_think_model_path,
    maybe_install_olmo_chat_template,
)
from verl_rlsd.ministral_tokenizer import load_eval_tokenizer

# Reuse dataset loading + grading from the vLLM eval entrypoint.
from eval_math_vllm_local import (
    _HAS_MATH_VERIFY,
    _apply_chat_prompt,
    _eval_output_paths,
    _is_gemma3_model,
    _load_resume_results,
    _math_user_suffix,
    _write_eval_checkpoints,
    default_data_root,
    extract_math_answer,
    extract_mcq_answer,
    grade_answer,
    load_examples,
    normalize_dataset_key,
    parse_pass_at_k,
    resolve_dataset_path,
    summarize_result_subset,
)


def _eos_token_ids(tokenizer: Any, model: Any) -> List[int]:
    """Prefer generation_config multi-EOS (Olmo: <|im_end|> + <|endoftext|>)."""
    ids: List[int] = []
    gen_cfg = getattr(model, "generation_config", None)
    raw = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
    if raw is None:
        raw = getattr(tokenizer, "eos_token_id", None)
    if isinstance(raw, int):
        ids = [raw]
    elif isinstance(raw, Sequence):
        ids = [int(x) for x in raw if x is not None]
    # Always include ChatML turn end if present.
    try:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0 and im_end not in ids:
            ids.append(im_end)
    except Exception:
        pass
    # Dedup preserve order
    out: List[int] = []
    for i in ids:
        if i not in out:
            out.append(i)
    return out or [tokenizer.eos_token_id]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode_completion(
    tokenizer: Any,
    token_ids: List[int],
    eos_token_id: List[int],
) -> tuple[str, int]:
    eos_set = set(eos_token_id)
    trimmed = list(token_ids)
    while trimmed and trimmed[-1] in eos_set:
        trimmed.pop()
    text = tokenizer.decode(trimmed, skip_special_tokens=True)
    return text, len(trimmed)


@torch.inference_mode()
def generate_n(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    num_return_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    eos_token_id: List[int],
    pad_token_id: int,
    seed: int,
) -> List[tuple[str, int]]:
    """Generate ``num_return_sequences`` completions for one prompt in a single generate call."""
    _set_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[-1])

    do_sample = temperature > 0
    gen_kwargs: Dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": max(1, int(num_return_sequences)),
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "use_cache": True,
    }
    if do_sample:
        gen_kwargs["temperature"] = max(temperature, 1e-5)
        gen_kwargs["top_p"] = top_p
        if top_k > 0:
            gen_kwargs["top_k"] = top_k

    out = model.generate(**gen_kwargs)
    results: List[tuple[str, int]] = []
    for i in range(out.shape[0]):
        new_ids = out[i, input_len:].tolist()
        results.append(_decode_completion(tokenizer, new_ids, eos_token_id))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Local math eval (HF transformers.generate)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--data-path", action="append", default=None)
    parser.add_argument("--data-format", type=str, default="auto")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1, help="-1 disables top-k")
    parser.add_argument("--val-n", type=int, default=8)
    parser.add_argument("--pass-at-k", type=str, default="1,4,8")
    parser.add_argument(
        "--generate-batch-size",
        type=int,
        default=8,
        help="num_return_sequences per generate call within one problem (chunk size over val-n).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--relaxed-answer-extraction", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        help="HF attn implementation (sdpa/eager/flash_attention_2)",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else default_data_root()
    load_queue: List[tuple[str, Optional[str]]] = []
    if args.dataset:
        for dn in args.dataset:
            p = resolve_dataset_path(dn, data_root, must_exist=True)
            load_queue.append((str(p), normalize_dataset_key(dn)))
    if args.data_path:
        for raw in args.data_path:
            load_queue.append((raw, None))
    if not load_queue:
        raise SystemExit("error: provide --dataset and/or --data-path")

    pass_at_k_list = parse_pass_at_k(args.pass_at_k)
    max_k = max(pass_at_k_list)
    gen_n = max(args.val_n, max_k)
    if gen_n != args.val_n:
        print(f"[eval] val-n {args.val_n} < max(pass-at-k)={max_k}; generating n={gen_n}", flush=True)

    if not _HAS_MATH_VERIFY:
        print("[warn] math_verify not installed; falling back to string equality", flush=True)

    limit = args.num_samples if args.num_samples > 0 else None
    resolved_paths: List[Path] = []
    examples: List[Dict[str, Any]] = []
    tag_counts: Dict[str, int] = {}
    for raw, tag_override in load_queue:
        data_path = Path(raw).expanduser().resolve()
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        resolved_paths.append(data_path)
        batch = load_examples(data_path, args.data_format, limit)
        base_tag = tag_override if tag_override is not None else data_path.stem
        tag_counts[base_tag] = tag_counts.get(base_tag, 0) + 1
        tag = base_tag if tag_counts[base_tag] == 1 else f"{base_tag}_{tag_counts[base_tag]}"
        for ex in batch:
            ex["dataset_tag"] = tag
            ex["dataset_path"] = str(data_path)
        examples.extend(batch)
        print(f"[eval] +{len(batch)} problems from {data_path} (tag={tag})", flush=True)

    if not examples:
        raise RuntimeError("No examples loaded")

    model_path = str(Path(args.model_path).expanduser().resolve())
    is_gemma3 = _is_gemma3_model(model_path)
    is_olmo = is_olmo_model_path(model_path)
    is_olmo_think = is_olmo_think_model_path(model_path)
    is_olmo_instruct = is_olmo_instruct_model_path(model_path)
    enable_thinking = bool(args.enable_thinking) and not is_gemma3 and not is_olmo_instruct
    if is_olmo_instruct and args.enable_thinking:
        print("[eval] Olmo-3-Instruct has no thinking mode; using instruct settings", flush=True)
    if is_olmo_think:
        print(
            "[eval] Olmo-3-Think: chat_template.jinja opens assistant with <think>",
            flush=True,
        )

    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else 32768

    print(f"[eval] backend=transformers total={len(examples)} thinking={enable_thinking}", flush=True)
    print(f"[eval] temp={temperature} top_p={top_p} top_k={top_k} max_new_tokens={max_new_tokens} n={gen_n}", flush=True)

    tokenizer = load_eval_tokenizer(model_path)
    maybe_install_olmo_chat_template(tokenizer, model_path=model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    print(f"[eval] loading HF model {model_path} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        attn_implementation=args.attn_implementation,
        device_map="auto",
    )
    model.eval()
    eos_ids = _eos_token_ids(tokenizer, model)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    print(f"[eval] eos_token_id={eos_ids} pad_token_id={pad_id}", flush=True)

    all_prompts: List[str] = []
    for ex in examples:
        eval_type = str(ex.get("eval_type", "boxed_math"))
        user_suffix = _math_user_suffix(eval_type, is_gemma3)
        messages = [{"role": "user", "content": ex["problem"] + user_suffix}]
        all_prompts.append(
            _apply_chat_prompt(
                tokenizer, messages, False if is_olmo else enable_thinking
            )
        )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path, results_path = _eval_output_paths(out_path)

    results: List[Dict[str, Any]] = []
    pass_at_k_counts: Dict[int, int] = {k: 0 for k in pass_at_k_list}
    formatted_total = 0
    total_solutions = 0
    total_correct = 0
    total_output_tokens = 0
    majority_correct = 0
    n_prompts_total = len(examples)

    work_examples = examples
    work_prompts = all_prompts
    if args.resume:
        resumed = _load_resume_results(results_path, gen_n)
        if resumed:
            done_ids = {str(r.get("problem_id")) for r in resumed}
            keep_idx = [i for i, ex in enumerate(examples) if str(ex["id"]) not in done_ids]
            results = list(resumed)
            # rebuild counters
            for r in results:
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
            work_examples = [examples[i] for i in keep_idx]
            work_prompts = [all_prompts[i] for i in keep_idx]
            print(
                f"[eval] resume: loaded {len(results)}; remaining {len(work_examples)}/{n_prompts_total}",
                flush=True,
            )

    gbs = max(1, int(args.generate_batch_size))
    print(
        f"[eval] generating {len(work_examples)} prompts x n={gen_n} "
        f"(HF num_return_sequences batch_size={gbs}) ...",
        flush=True,
    )

    for ex, prompt in tqdm(
        list(zip(work_examples, work_prompts)),
        desc="problems",
        dynamic_ncols=True,
    ):
        gt = ex["ground_truth"]
        generations: List[str] = []
        preds: List[str] = []
        correct_flags: List[bool] = []
        formatted_flags: List[bool] = []
        token_counts: List[int] = []

        sample_i = 0
        while sample_i < gen_n:
            n_this = min(gbs, gen_n - sample_i)
            chunk = generate_n(
                model,
                tokenizer,
                prompt,
                num_return_sequences=n_this,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos_ids,
                pad_token_id=pad_id,
                seed=args.seed + sample_i,
            )
            sample_i += n_this

            for gen, n_tokens in chunk:
                generations.append(gen)
                token_counts.append(n_tokens)
                total_output_tokens += n_tokens

                eval_type = str(ex.get("eval_type", "boxed_math"))
                if eval_type == "mcq":
                    pred = extract_mcq_answer(gen)
                    formatted = pred is not None
                    gt_choice = str(ex.get("ground_truth_choice", ex["ground_truth"])).upper().strip()
                    ok = bool(pred is not None and pred.upper() == gt_choice)
                else:
                    pred = extract_math_answer(gen, relaxed=args.relaxed_answer_extraction)
                    formatted = pred is not None
                    ok = grade_answer(pred, gt)

                if pred is None:
                    preds.append("[no answer]" if args.relaxed_answer_extraction else "[no boxed]")
                else:
                    preds.append(pred)
                correct_flags.append(ok)
                formatted_flags.append(formatted)
                total_solutions += 1
                if formatted:
                    formatted_total += 1
                if ok:
                    total_correct += 1

        pass_at_k_problem: Dict[str, bool] = {}
        for k in pass_at_k_list:
            ok_k = any(correct_flags[:k])
            pass_at_k_problem[str(k)] = ok_k
            if ok_k:
                pass_at_k_counts[k] += 1

        maj_ok = False
        fpreds = [p for p, f in zip(preds, formatted_flags) if f]
        if fpreds:
            top = Counter(fpreds).most_common(1)[0][0]
            maj_ok = grade_answer(top, gt)
        if maj_ok:
            majority_correct += 1

        results.append(
            {
                "dataset_tag": ex.get("dataset_tag", ""),
                "dataset_path": ex.get("dataset_path", ""),
                "category": ex.get("category", ""),
                "problem_id": ex["id"],
                "problem": ex["problem"],
                "ground_truth": gt,
                "gen_n": gen_n,
                "pass_at_k": pass_at_k_problem,
                "generations": [
                    {
                        "predicted_answer": p,
                        "full_generation": g,
                        "num_tokens": n,
                        "correct": c,
                        "formatted": f,
                    }
                    for p, g, n, c, f in zip(
                        preds, generations, token_counts, correct_flags, formatted_flags
                    )
                ],
                "avg_output_tokens": sum(token_counts) / len(token_counts) if token_counts else 0.0,
                "num_correct": sum(correct_flags),
                "pass_at_gen_n": bool(any(correct_flags)),
                "majority_vote_correct": maj_ok,
                "predicted_answer": preds[0],
                "full_generation": generations[0],
                "correct": correct_flags[0],
                "formatted": formatted_flags[0],
            }
        )

        processed = len(results)
        pass_at_k_summary = {
            str(k): {
                "count": pass_at_k_counts[k],
                "total": processed,
                "pct": 100.0 * pass_at_k_counts[k] / processed if processed else 0.0,
            }
            for k in pass_at_k_list
        }
        by_tag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in results:
            by_tag[str(r.get("dataset_tag", ""))].append(r)
        metrics_by_dataset = {
            tag: {"dataset_path": sub[0].get("dataset_path", ""), **summarize_result_subset(sub, pass_at_k_list, gen_n)}
            for tag, sub in sorted(by_tag.items())
        }
        by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in results:
            cat = str(r.get("category", "")).strip() or "__uncategorized__"
            by_category[cat].append(r)
        metrics_by_category = {
            cat: summarize_result_subset(sub, pass_at_k_list, gen_n)
            for cat, sub in sorted(by_category.items())
        }
        summary = {
            "model_path": model_path,
            "backend": "transformers",
            "attn_implementation": args.attn_implementation,
            "dtype": args.dtype,
            "eos_token_id": eos_ids,
            "data_root": str(data_root),
            "data_paths": [str(p) for p in resolved_paths],
            "dataset_args": list(args.dataset) if args.dataset else [],
            "data_format": args.data_format,
            "enable_thinking": enable_thinking,
            "relaxed_answer_extraction": args.relaxed_answer_extraction,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "val_n_requested": args.val_n,
            "gen_n": gen_n,
            "generate_batch_size": gbs,
            "pass_at_k_list": pass_at_k_list,
            "pass_at_k": pass_at_k_summary,
            "avg1_pct": pass_at_k_summary.get("1", {}).get("pct", 0.0),
            "avg16_pct": 100.0 * total_correct / total_solutions if total_solutions else 0.0,
            "metrics_by_dataset": metrics_by_dataset,
            "metrics_by_category": metrics_by_category,
            "num_problems": processed,
            "num_problems_total": n_prompts_total,
            "total_solutions": total_solutions,
            "average_correct_pct": 100.0 * total_correct / total_solutions if total_solutions else 0.0,
            "majority_vote_pct": 100.0 * majority_correct / processed if processed else 0.0,
            "format_rate_pct": 100.0 * formatted_total / total_solutions if total_solutions else 0.0,
            "avg_output_tokens_mean": total_output_tokens / total_solutions if total_solutions else 0.0,
            "math_verify": _HAS_MATH_VERIFY,
            "streaming_write": True,
            "metrics_json": str(metrics_path),
            "results_json": str(results_path),
            "results": results,
        }
        _write_eval_checkpoints(
            metrics_path=metrics_path,
            results_path=results_path,
            summary=summary,
            results=results,
            processed=processed,
            n_prompts=n_prompts_total,
        )

    processed = len(results)
    print("=" * 60, flush=True)
    print(f"Problems: {processed}/{n_prompts_total}", flush=True)
    for k in pass_at_k_list:
        c = pass_at_k_counts[k]
        pct = 100.0 * c / processed if processed else 0.0
        print(f"pass@{k}: {c}/{processed} = {pct:.2f}%", flush=True)
    print(
        f"format_rate: {100.0 * formatted_total / total_solutions if total_solutions else 0.0:.2f}%",
        flush=True,
    )
    print(
        f"avg_output_tokens: {total_output_tokens / total_solutions if total_solutions else 0.0:.1f}",
        flush=True,
    )
    print(f"Wrote metrics {metrics_path}", flush=True)
    print(f"Wrote results {results_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
