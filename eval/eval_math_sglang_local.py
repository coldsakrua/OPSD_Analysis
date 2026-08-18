#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Math eval with SGLang Engine (Olmo-3 native path; use --attention-backend triton if no fa3)."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from transformers import AutoTokenizer

from verl_rlsd.olmo_chat_template import (
    is_olmo_instruct_model_path,
    is_olmo_model_path,
    is_olmo_think_model_path,
    maybe_install_olmo_chat_template,
)

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


def _normalize_generate_outputs(raw: Any, n_prompts: int, gen_n: int) -> List[List[Dict[str, Any]]]:
    """Map Engine.generate return value to [prompt][sample] dicts with keys text/meta_info."""
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise RuntimeError(f"unexpected SGLang generate return type: {type(raw)}")

    # Case A: flat list length == n_prompts * gen_n
    if len(raw) == n_prompts * gen_n:
        grouped: List[List[Dict[str, Any]]] = []
        for i in range(n_prompts):
            grouped.append(raw[i * gen_n : (i + 1) * gen_n])
        return grouped

    # Case B: one entry per prompt; text is list when n>1
    if len(raw) == n_prompts:
        grouped = []
        for item in raw:
            texts = item.get("text")
            meta = item.get("meta_info") or {}
            if isinstance(texts, list):
                samples = []
                for t in texts:
                    samples.append({"text": t, "meta_info": meta})
                if len(samples) < gen_n:
                    raise RuntimeError(f"expected {gen_n} texts, got {len(samples)}")
                grouped.append(samples[:gen_n])
            else:
                if gen_n != 1:
                    raise RuntimeError(f"expected n={gen_n} samples, got single text")
                grouped.append([item])
        return grouped

    raise RuntimeError(
        f"cannot reshape SGLang outputs: len={len(raw)} n_prompts={n_prompts} gen_n={gen_n}"
    )


def _completion_tokens(sample: Dict[str, Any], text: str, tokenizer: Any) -> int:
    meta = sample.get("meta_info") or {}
    for key in ("completion_tokens", "completion_tokens_count", "output_tokens"):
        if key in meta and meta[key] is not None:
            return int(meta[key])
    usage = meta.get("usage") or {}
    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
        return int(usage["completion_tokens"])
    return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Local math eval (SGLang Engine)")
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
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--val-n", type=int, default=8)
    parser.add_argument("--pass-at-k", type=str, default="1,4,8")
    parser.add_argument(
        "--generate-batch-size",
        type=int,
        default=8,
        help="Number of prompts per Engine.generate call.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--relaxed-answer-extraction", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="triton",
        help="SGLang attention backend. Olmo3: prefer triton when fa3 is unavailable.",
    )
    parser.add_argument(
        "--sampling-backend",
        type=str,
        default="pytorch",
        choices=["pytorch", "flashinfer", "ascend"],
        help=(
            "SGLang sampling backend. Default pytorch avoids FlashInfer JIT "
            "(fails on nodes with old /usr/local/cuda nvcc)."
        ),
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.80)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=40960)
    parser.add_argument(
        "--disable-multimodal",
        action="store_true",
        default=False,
        help=(
            "Pass enable_multimodal=False to SGLang. "
            "NOTE: currently breaks Qwen3.5 on sglang 0.5.10 (mrope positions=None); "
            "leave unset for Qwen3.5 text eval."
        ),
    )
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default="",
        help="SGLang reasoning parser name (e.g. qwen3). Empty disables.",
    )
    parser.add_argument(
        "--disable-hybrid-swa-memory",
        action="store_true",
        default=True,
        help="Pass disable_hybrid_swa_memory to Engine (needed for some models).",
    )
    parser.add_argument(
        "--enable-hybrid-swa-memory",
        dest="disable_hybrid_swa_memory",
        action="store_false",
    )
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
    # Xiaomi MiMo recommends an empty system prompt (avoid default "helpful assistant").
    is_mimo = "mimo" in Path(model_path).name.lower()
    # Olmo-Instruct has no think mode; Olmo-Think uses chat_template.jinja that
    # already opens with <think> (no HF enable_thinking switch).
    enable_thinking = bool(args.enable_thinking) and not is_gemma3 and not is_olmo_instruct
    if is_olmo_instruct and args.enable_thinking:
        print("[eval] Olmo-3-Instruct has no thinking mode; using instruct settings", flush=True)
    if is_olmo_think:
        print(
            "[eval] Olmo-3-Think: using model chat_template.jinja "
            "(generation prompt ends with <think>); "
            f"official sampling temp=0.6 top_p=0.95 max_tokens=32768",
            flush=True,
        )
    if is_mimo:
        # Official MiMo-7B-RL eval (ModelScope/HF/paper): temp=0.6, top_p=0.95,
        # max_tokens=32768 for math; empty system prompt.
        # https://www.modelscope.cn/models/XiaomiMiMo/MiMo-7B-RL
        print(
            "[eval] MiMo: empty system prompt; official sampling "
            f"temp={args.temperature} top_p={args.top_p} max_tokens={args.max_new_tokens}",
            flush=True,
        )

    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else 32768
    gbs = max(1, int(args.generate_batch_size))

    print(
        f"[eval] backend=sglang attention_backend={args.attention_backend} "
        f"sampling_backend={args.sampling_backend} "
        f"disable_multimodal={args.disable_multimodal} "
        f"reasoning_parser={args.reasoning_parser or 'none'} "
        f"total={len(examples)} thinking={enable_thinking}",
        flush=True,
    )
    print(
        f"[eval] temp={temperature} top_p={top_p} top_k={top_k} "
        f"min_p={args.min_p} presence_penalty={args.presence_penalty} "
        f"max_new_tokens={max_new_tokens} n={gen_n} prompt_batch={gbs}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    maybe_install_olmo_chat_template(tokenizer, model_path=model_path)
    stop_ids = _stop_token_ids(tokenizer)
    print(f"[eval] stop_token_ids={stop_ids}", flush=True)
    # Smoke-check Think template opens the assistant turn with <think>.
    if is_olmo_think:
        _probe = _apply_chat_prompt(
            tokenizer, [{"role": "user", "content": "ping"}], enable_thinking=False
        )
        if not _probe.rstrip().endswith("<think>"):
            print(
                "[warn] Olmo-Think chat template did not end with <think>; "
                f"prompt_tail={_probe[-120:]!r}",
                flush=True,
            )
        else:
            print("[eval] Think chat template OK (assistant opens with <think>)", flush=True)

    all_prompts: List[str] = []
    for ex in examples:
        eval_type = str(ex.get("eval_type", "boxed_math"))
        user_suffix = _math_user_suffix(eval_type, is_gemma3)
        messages: List[Dict[str, str]] = []
        if is_mimo:
            messages.append({"role": "system", "content": ""})
        messages.append({"role": "user", "content": ex["problem"] + user_suffix})
        # Olmo templates ignore enable_thinking; pass False to avoid unused jinja vars.
        all_prompts.append(
            _apply_chat_prompt(
                tokenizer, messages, False if is_olmo else enable_thinking
            )
        )

    # SGLang rejects when (prompt_len + max_new_tokens) >= context_length
    # (see tokenizer_manager._validate_one_request). Official math evals ask for
    # max_tokens=32768 with the same model context; clamp to remaining budget - 1.
    prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in all_prompts]
    max_prompt_tokens = max(prompt_lens) if prompt_lens else 0
    context_length = int(args.context_length)
    requested_max_new = max_new_tokens
    # Must satisfy: max_new + max_prompt < context_length
    max_allowed_new = max(1, context_length - max_prompt_tokens - 1)
    if max_new_tokens > max_allowed_new:
        max_new_tokens = max_allowed_new
        print(
            f"[eval] clamp max_new_tokens {requested_max_new} -> {max_new_tokens} "
            f"(max_prompt={max_prompt_tokens} + gen must be < context_length={context_length})",
            flush=True,
        )
    else:
        print(
            f"[eval] context ok: max_prompt={max_prompt_tokens} + "
            f"max_new_tokens={max_new_tokens} < context_length={context_length}",
            flush=True,
        )
    # #region agent log
    try:
        import time as _agent_time

        with open(
            "/gpfs/share/home/2501210611/opsd_analysis/.cursor/debug-71bbf7.log",
            "a",
            encoding="utf-8",
        ) as _agent_f:
            _agent_f.write(
                json.dumps(
                    {
                        "sessionId": "71bbf7",
                        "runId": "post-fix",
                        "hypothesisId": "A",
                        "location": "eval_math_sglang_local.py:clamp",
                        "message": "context clamp decision",
                        "data": {
                            "requested_max_new": requested_max_new,
                            "max_prompt_tokens": max_prompt_tokens,
                            "context_length": context_length,
                            "max_allowed_new": max_allowed_new,
                            "final_max_new_tokens": max_new_tokens,
                            "sum_prompt_plus_new": max_prompt_tokens + max_new_tokens,
                            "sglang_would_reject": (
                                max_prompt_tokens + max_new_tokens
                            )
                            >= context_length,
                            "is_mimo": is_mimo,
                            "temperature": temperature,
                            "top_p": top_p,
                        },
                        "timestamp": int(_agent_time.time() * 1000),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion

    from sglang import Engine

    print(f"[eval] launching SGLang Engine for {model_path} ...", flush=True)
    engine_kwargs: Dict[str, Any] = {
        "model_path": model_path,
        "tokenizer_path": model_path,
        "trust_remote_code": True,
        "attention_backend": args.attention_backend,
        "sampling_backend": args.sampling_backend,
        "mem_fraction_static": args.mem_fraction_static,
        "tp_size": args.tp_size,
        "context_length": context_length,
        "log_level": "warning",  # suppress Prefill/Decode batch info spam
    }
    if args.disable_hybrid_swa_memory:
        engine_kwargs["disable_hybrid_swa_memory"] = True
    if args.disable_multimodal:
        # Text-only eval: skip multimodal path / VLM mem reserve (not SGLang language_only EPD).
        engine_kwargs["enable_multimodal"] = False
    if args.reasoning_parser:
        engine_kwargs["reasoning_parser"] = args.reasoning_parser
    llm = Engine(**engine_kwargs)

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

    print(
        f"[eval] generating {len(work_examples)} prompts x n={gen_n} "
        f"(SGLang prompt_batch={gbs}) ...",
        flush=True,
    )

    sampling_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "n": gen_n,
        "sampling_seed": args.seed,
        "skip_special_tokens": True,
    }
    if top_k > 0:
        sampling_params["top_k"] = top_k
    if args.min_p and args.min_p > 0:
        sampling_params["min_p"] = args.min_p
    if args.presence_penalty != 0.0:
        sampling_params["presence_penalty"] = args.presence_penalty
    if stop_ids:
        sampling_params["stop_token_ids"] = stop_ids

    def flush_summary() -> None:
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
            tag: {
                "dataset_path": sub[0].get("dataset_path", ""),
                **summarize_result_subset(sub, pass_at_k_list, gen_n),
            }
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
            "backend": "sglang",
            "attention_backend": args.attention_backend,
            "stop_token_ids": stop_ids,
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
            # avg16_pct: legacy alias; avg{gen_n}_pct is the canonical mean-over-n accuracy.
            "avg16_pct": 100.0 * total_correct / total_solutions if total_solutions else 0.0,
            f"avg{gen_n}_pct": 100.0 * total_correct / total_solutions if total_solutions else 0.0,
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

    try:
        for start in tqdm(range(0, len(work_examples), gbs), desc="prompt_batches", dynamic_ncols=True):
            end = min(start + gbs, len(work_examples))
            chunk_ex = work_examples[start:end]
            chunk_prompts = work_prompts[start:end]
            # #region agent log
            try:
                import time as _agent_time

                _chunk_lens = [
                    len(tokenizer.encode(p, add_special_tokens=False))
                    for p in chunk_prompts
                ]
                with open(
                    "/gpfs/share/home/2501210611/opsd_analysis/.cursor/debug-71bbf7.log",
                    "a",
                    encoding="utf-8",
                ) as _agent_f:
                    _agent_f.write(
                        json.dumps(
                            {
                                "sessionId": "71bbf7",
                                "runId": "post-fix",
                                "hypothesisId": "A",
                                "location": "eval_math_sglang_local.py:generate",
                                "message": "pre-generate batch token budget",
                                "data": {
                                    "batch_start": start,
                                    "batch_end": end,
                                    "chunk_prompt_lens": _chunk_lens,
                                    "max_new_tokens": sampling_params.get(
                                        "max_new_tokens"
                                    ),
                                    "worst_sum": (
                                        max(_chunk_lens) if _chunk_lens else 0
                                    )
                                    + int(sampling_params.get("max_new_tokens") or 0),
                                    "context_length": context_length,
                                    "would_reject": any(
                                        (pl + int(sampling_params.get("max_new_tokens") or 0))
                                        >= context_length
                                        for pl in _chunk_lens
                                    ),
                                },
                                "timestamp": int(_agent_time.time() * 1000),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            raw = llm.generate(chunk_prompts, sampling_params=sampling_params)
            grouped = _normalize_generate_outputs(raw, len(chunk_ex), gen_n)

            for ex, samples in zip(chunk_ex, grouped):
                gt = ex["ground_truth"]
                generations: List[str] = []
                preds: List[str] = []
                correct_flags: List[bool] = []
                formatted_flags: List[bool] = []
                token_counts: List[int] = []

                for sample in samples:
                    gen = sample.get("text") or ""
                    if isinstance(gen, list):
                        gen = gen[0] if gen else ""
                    n_tokens = _completion_tokens(sample, gen, tokenizer)
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
            flush_summary()
    finally:
        try:
            llm.shutdown()
        except Exception:
            pass

    processed = len(results)
    avg_n_pct = 100.0 * total_correct / total_solutions if total_solutions else 0.0
    print("=" * 60, flush=True)
    print(f"Problems: {processed}/{n_prompts_total}", flush=True)
    for k in pass_at_k_list:
        c = pass_at_k_counts[k]
        pct = 100.0 * c / processed if processed else 0.0
        print(f"pass@{k}: {c}/{processed} = {pct:.2f}%", flush=True)
    print(f"avg{gen_n}: {avg_n_pct:.2f}%", flush=True)
    print(
        f"majority_vote: {100.0 * majority_correct / processed if processed else 0.0:.2f}%",
        flush=True,
    )
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
