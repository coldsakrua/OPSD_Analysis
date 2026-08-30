#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continue truncated eval generations up to target max_new_tokens and rewrite json/logs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# Reuse eval helpers.
from eval_math_vllm_local import (  # type: ignore
    DEFAULT_TOKEN_BUDGETS,
    _apply_chat_prompt,
    _completion_token_count,
    _is_gemma3_model,
    _math_user_suffix,
    _stop_token_ids,
    append_postprocess_metrics_to_out,
    build_llm,
    compute_ap_lne_metrics,
    compute_token_budget_metrics,
    extract_math_answer,
    extract_mcq_answer,
    format_ap_lne_lines,
    format_token_budget_lines,
    grade_answer,
    load_eval_tokenizer,
    max_seq_len_from_model_config,
    maybe_install_olmo_chat_template,
    summarize_result_subset,
)

EXTEND_BUDGETS = sorted(set(DEFAULT_TOKEN_BUDGETS + [38912]))


def _try_remap_original_log(metrics_path: Path, eval_tag: str) -> Optional[Path]:
    """Locate the original eval .out that wrote this metrics file (exclude extend_38912)."""
    log_root = Path("log/eval")
    if not log_root.is_dir():
        return None
    needle = str(metrics_path.resolve())
    try:
        proc = subprocess.run(
            ["rg", "-l", "--glob", "*.out", "-F", needle, str(log_root)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception:
        return None
    hits: List[Path] = []
    for line in proc.stdout.splitlines():
        p = Path(line.strip())
        if p.is_file() and "extend_38912" not in str(p):
            hits.append(p)
    if not hits and eval_tag:
        needle2 = f"eval_outputs/{eval_tag}/"
        try:
            proc2 = subprocess.run(
                ["rg", "-l", "--glob", "*.out", "-F", needle2, str(log_root)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except Exception:
            proc2 = None
        if proc2 is not None:
            mname = metrics_path.name
            for line in proc2.stdout.splitlines():
                p = Path(line.strip())
                if not p.is_file() or "extend_38912" in str(p):
                    continue
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if mname in txt or needle in txt:
                    hits.append(p)
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def batch_size_for_model(model_path: str, override: int = 0) -> int:
    """Pick generate batch size by model scale for ~32k-prefix continues on A800 80G."""
    if override and override > 0:
        return int(override)
    p = model_path.lower().replace("-", "_").replace(".", "p")
    # Larger models / longer KV: smaller batch. Concurrent room at 40k is ~15 for 1.7B.
    if "8b" in p or "/8b/" in p or "_8b_" in p:
        return 4
    if "4b" in p or "/4b/" in p or "_4b_" in p:
        return 8
    if "0p6b" in p or "06b" in p or "0.6b" in model_path.lower():
        return 12
    if "1p7b" in p or "1.7b" in model_path.lower() or "1_7b" in p:
        return 12
    return 8


def _atomic_write_json(path: Path, obj: Any, *, indent: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if indent is not None:
        text = json.dumps(obj, ensure_ascii=False, indent=indent)
    else:
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _regrade_generation(
    text: str,
    gt: str,
    *,
    eval_type: str,
    gt_choice: str,
    relaxed: bool,
) -> Tuple[Optional[str], bool, bool]:
    if eval_type == "mcq":
        pred = extract_mcq_answer(text)
        formatted = pred is not None
        ok = bool(pred is not None and pred.upper() == gt_choice.upper().strip())
        return pred, ok, formatted
    pred = extract_math_answer(text, relaxed=relaxed)
    formatted = pred is not None
    ok = grade_answer(pred, gt)
    return pred, ok, formatted


def _refresh_problem_row(row: Dict[str, Any], pass_at_k_list: List[int], relaxed: bool) -> None:
    gt = str(row.get("ground_truth", ""))
    eval_type = str(row.get("eval_type", "boxed_math"))
    gt_choice = str(row.get("ground_truth_choice", gt))
    gens = row.get("generations") or []
    preds: List[str] = []
    correct_flags: List[bool] = []
    formatted_flags: List[bool] = []
    token_counts: List[int] = []
    for g in gens:
        text = g.get("full_generation") or ""
        pred, ok, formatted = _regrade_generation(
            text, gt, eval_type=eval_type, gt_choice=gt_choice, relaxed=relaxed
        )
        if pred is None:
            pred_s = "[no boxed]" if not relaxed else "[no answer]"
        else:
            pred_s = pred
        g["predicted_answer"] = pred_s
        g["correct"] = ok
        g["formatted"] = formatted
        preds.append(pred_s)
        correct_flags.append(ok)
        formatted_flags.append(formatted)
        token_counts.append(int(g.get("num_tokens") or 0))

    pass_at_k_problem: Dict[str, bool] = {}
    for k in pass_at_k_list:
        pass_at_k_problem[str(k)] = bool(any(correct_flags[:k]))
    row["pass_at_k"] = pass_at_k_problem
    row["avg_output_tokens"] = sum(token_counts) / len(token_counts) if token_counts else 0.0
    row["num_correct"] = sum(correct_flags)
    row["pass_at_gen_n"] = bool(any(correct_flags))
    fpreds = [p for p, f in zip(preds, formatted_flags) if f]
    maj_ok = False
    if fpreds:
        top = Counter(fpreds).most_common(1)[0][0]
        maj_ok = grade_answer(top, gt)
    row["majority_vote_correct"] = maj_ok
    if gens:
        row["predicted_answer"] = preds[0]
        row["full_generation"] = gens[0].get("full_generation", "")
        row["correct"] = correct_flags[0]
        row["formatted"] = formatted_flags[0]


def _rebuild_metrics(
    metrics: Dict[str, Any],
    results: List[Dict[str, Any]],
    *,
    target_max_new: int,
    tokenizer: Any,
    pass_at_k_list: List[int],
    relaxed: bool,
    extended_count: int,
) -> Dict[str, Any]:
    gen_n = int(metrics.get("gen_n") or (len(results[0]["generations"]) if results else 8))
    by_tag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_tag[str(r.get("dataset_tag", ""))].append(r)
    metrics_by_dataset: Dict[str, Any] = {}
    for tag, sub in sorted(by_tag.items(), key=lambda x: x[0]):
        path0 = sub[0].get("dataset_path", "") if sub else ""
        metrics_by_dataset[tag] = {
            "dataset_path": path0,
            **summarize_result_subset(sub, pass_at_k_list, gen_n),
        }
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        cat = str(r.get("category", "")).strip() or "__uncategorized__"
        by_category[cat].append(r)
    metrics_by_category = {
        cat: summarize_result_subset(sub, pass_at_k_list, gen_n)
        for cat, sub in sorted(by_category.items(), key=lambda x: x[0])
    }
    combined = summarize_result_subset(results, pass_at_k_list, gen_n)
    token_budget = compute_token_budget_metrics(
        results, tokenizer, pass_at_k_list, budgets=EXTEND_BUDGETS, relaxed=relaxed
    )
    ap_lne = compute_ap_lne_metrics(results)

    out = dict(metrics)
    out["max_new_tokens"] = target_max_new
    out["pre_extend_max_new_tokens"] = int(metrics.get("pre_extend_max_new_tokens") or metrics.get("max_new_tokens") or 32768)
    out["extended_to"] = target_max_new
    out["extended_generations"] = extended_count
    out["pass_at_k"] = combined["pass_at_k"]
    out["avg1_pct"] = combined["avg1_pct"]
    out["avg16_pct"] = combined["avg16_pct"]
    out["metrics_by_dataset"] = metrics_by_dataset
    out["metrics_by_category"] = metrics_by_category
    out["num_problems"] = combined["num_problems"]
    out["average_correct_pct"] = combined["average_correct_pct"]
    out["majority_vote_pct"] = combined["majority_vote_pct"]
    out["format_rate_pct"] = combined["format_rate_pct"]
    out["avg_output_tokens_mean"] = combined["avg_output_tokens_mean"]
    out["token_budget_metrics"] = token_budget
    out["ap_lne_metrics"] = ap_lne
    out["partial_only"] = False
    return out


def _append_extend_log(
    log_path: Path,
    *,
    job: Dict[str, Any],
    before: Dict[str, Any],
    after: Dict[str, Any],
    extended_count: int,
    elapsed_s: float,
    pass_at_k_list: List[int],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "",
        "=" * 60,
        "[EXTEND_TO_38912]",
        f"metrics_json={job['metrics_json']}",
        f"results_json={job['results_json']}",
        f"model_path={job['model_path']}",
        f"old_max_new={job['old_max_new']} -> target_max_new={job['target_max_new']}",
        f"extended_generations={extended_count}",
        f"elapsed_sec={elapsed_s:.1f}",
        f"before Pass@8={before.get('pass_at_k', {}).get('8', {}).get('pct', 0):.2f}% "
        f"Avg16={before.get('avg16_pct', 0):.2f}% "
        f"Len={before.get('avg_output_tokens_mean', 0):.1f}",
        f"after  Pass@8={after.get('pass_at_k', {}).get('8', {}).get('pct', 0):.2f}% "
        f"Avg16={after.get('avg16_pct', 0):.2f}% "
        f"Len={after.get('avg_output_tokens_mean', 0):.1f}",
    ]
    for k in pass_at_k_list:
        s = after.get("pass_at_k", {}).get(str(k), {})
        lines.append(
            f"  Pass@{k}: {s.get('pct', 0):.2f}% ({s.get('count', 0)}/{s.get('total', 0)})"
        )
    lines.append(f"Avg correct / sample: {after.get('average_correct_pct', 0):.2f}%")
    lines.append(f"Majority vote: {after.get('majority_vote_pct', 0):.2f}%")
    lines.append(f"Boxed format rate: {after.get('format_rate_pct', 0):.2f}%")
    lines.append(f"Avg output tokens: {after.get('avg_output_tokens_mean', 0):.1f}")
    lines.append("=" * 60)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _destroy_llm(llm: Any) -> None:
    """Best-effort teardown so a second model can load in the same process.

    Multi-model shards previously OOM'd on the 2nd LLM() because vLLM/NCCL
    state was left behind after a plain ``del llm``.
    """
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None:
            executor = getattr(engine, "model_executor", None)
            if executor is not None:
                shutdown = getattr(executor, "shutdown", None)
                if callable(shutdown):
                    shutdown()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _chat_prefix(tokenizer: Any, problem: str, enable_thinking: bool, model_path: str) -> str:
    is_gemma3 = _is_gemma3_model(model_path)
    user_suffix = _math_user_suffix("boxed_math", is_gemma3)
    messages = [{"role": "user", "content": problem + user_suffix}]
    return _apply_chat_prompt(tokenizer, messages, enable_thinking)


def extend_job(
    job: Dict[str, Any],
    llm: Any,
    tokenizer: Any,
    *,
    generate_batch_size: int,
    tp: int,
    gpu_mem: float,
) -> int:
    metrics_path = Path(job["metrics_json"])
    results_path = Path(job["results_json"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    results_doc = json.loads(results_path.read_text(encoding="utf-8"))
    results: List[Dict[str, Any]] = results_doc.get("results") or []
    old_max = int(job["old_max_new"])
    target = int(job["target_max_new"])
    max_model_len = int(job.get("max_model_len") or 40960)
    allow_long = bool(job.get("allow_long_max_model_len"))
    if allow_long:
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    else:
        cfg_max = max_seq_len_from_model_config(job.get("vllm_base_model_path") or job["model_path"])
        if cfg_max is not None:
            max_model_len = min(max_model_len, cfg_max)
    trunc_slack = int(job.get("trunc_slack") or 0)
    trunc_thr = max(0, old_max - trunc_slack)
    pass_at_k_list = [int(x) for x in (job.get("pass_at_k_list") or [1, 4, 8])]
    relaxed = bool(job.get("relaxed_answer_extraction", False))
    enable_thinking = bool(job.get("enable_thinking", True))
    model_path = job["model_path"]

    before_summary = {
        "pass_at_k": metrics.get("pass_at_k"),
        "avg16_pct": metrics.get("avg16_pct"),
        "avg_output_tokens_mean": metrics.get("avg_output_tokens_mean"),
        "average_correct_pct": metrics.get("average_correct_pct"),
        "majority_vote_pct": metrics.get("majority_vote_pct"),
        "format_rate_pct": metrics.get("format_rate_pct"),
    }

    # Collect truncated (row_idx, gen_idx).
    work: List[Tuple[int, int, int, str]] = []
    for ri, row in enumerate(results):
        problem = str(row.get("problem") or "")
        prefix = _chat_prefix(tokenizer, problem, enable_thinking, model_path)
        for gi, g in enumerate(row.get("generations") or []):
            # Skip already-continued samples (even if they stopped via EOS below target).
            if int(g.get("extended_from") or 0) > 0:
                continue
            n = int(g.get("num_tokens") or 0)
            # trunc_slack: base models hit context wall slightly below max_new (prompt eats budget).
            if n >= trunc_thr and n < target:
                text = g.get("full_generation") or ""
                work.append((ri, gi, n, prefix + text))

    if not work:
        print(f"[extend] skip (no trunc): {metrics_path.name}", flush=True)
        return 0

    from vllm import SamplingParams

    stop_ids = _stop_token_ids(tokenizer)
    t0 = time.time()
    extended = 0
    gbs = max(1, generate_batch_size)

    for start in tqdm(range(0, len(work), gbs), desc=f"extend:{Path(job['eval_tag'])}/{job['dataset']}"):
        chunk = work[start : start + gbs]
        prompts: List[str] = []
        sps: List[Any] = []
        meta: List[Tuple[int, int, int]] = []
        for ri, gi, n_old, cont_prompt in chunk:
            prompt_len = len(tokenizer.encode(cont_prompt, add_special_tokens=False))
            room = max(1, max_model_len - prompt_len)
            need = max(1, target - n_old)
            max_tokens = min(room, need)
            prompts.append(cont_prompt)
            sp_kw: Dict[str, Any] = {
                "temperature": float(job.get("temperature", 0.6)),
                "top_p": float(job.get("top_p", 0.95)),
                "min_p": float(job.get("min_p", 0.0)),
                "max_tokens": max_tokens,
                "n": 1,
                "seed": int(job.get("seed", 42)),
            }
            top_k = int(job.get("top_k", 20))
            if top_k > 0:
                sp_kw["top_k"] = top_k
            pp = float(job.get("presence_penalty", 0.0))
            if pp:
                sp_kw["presence_penalty"] = pp
            if stop_ids:
                sp_kw["stop_token_ids"] = stop_ids
            sps.append(SamplingParams(**sp_kw))
            meta.append((ri, gi, n_old))

        outputs = llm.generate(prompts, sps, use_tqdm=False)
        for (ri, gi, n_old), out in zip(meta, outputs):
            o = out.outputs[0]
            cont = o.text or ""
            g = results[ri]["generations"][gi]
            old_text = g.get("full_generation") or ""
            new_text = old_text + cont
            n_new = _completion_token_count(new_text, getattr(o, "token_ids", None), tokenizer)
            # Prefer old+new token accounting when continuation token_ids are reliable.
            cont_n = _completion_token_count(cont, getattr(o, "token_ids", None), tokenizer)
            if cont_n > 0:
                n_new = n_old + cont_n
            g["full_generation"] = new_text
            g["num_tokens"] = int(min(n_new, target)) if n_new > target else int(n_new)
            g["extended_from"] = int(n_old)
            g["extended_target"] = int(target)
            extended += 1

    for row in results:
        _refresh_problem_row(row, pass_at_k_list, relaxed)

    after_metrics = _rebuild_metrics(
        metrics,
        results,
        target_max_new=target,
        tokenizer=tokenizer,
        pass_at_k_list=pass_at_k_list,
        relaxed=relaxed,
        extended_count=extended,
    )
    results_doc["results"] = results
    results_doc["partial_only"] = False
    results_doc["results_count"] = len(results)
    results_doc["metrics_json"] = str(metrics_path)
    after_metrics["results_json"] = str(results_path)
    after_metrics["metrics_json"] = str(metrics_path)

    _atomic_write_json(results_path, results_doc, indent=None)
    _atomic_write_json(metrics_path, after_metrics, indent=2)

    elapsed = time.time() - t0
    log_path = Path(job["log_out"])
    # If manifest fell back to extend_38912/*_extend.out, try to locate the real eval .out.
    if "extend_38912" in str(log_path) and str(log_path).endswith("_extend.out"):
        remapped = _try_remap_original_log(metrics_path, str(job.get("eval_tag") or ""))
        if remapped is not None:
            print(f"[extend] remapped log_out -> {remapped}", flush=True)
            log_path = remapped
            job["log_out"] = str(remapped)
    _append_extend_log(
        log_path,
        job=job,
        before=before_summary,
        after=after_metrics,
        extended_count=extended,
        elapsed_s=elapsed,
        pass_at_k_list=pass_at_k_list,
    )
    try:
        append_postprocess_metrics_to_out(
            log_path,
            results_path,
            metrics_path,
            budgets=EXTEND_BUDGETS,
            replace=True,
        )
    except Exception as e:
        print(f"[extend] warn: token-budget log refresh failed: {e}", flush=True)

    print(
        f"[extend] done {job['eval_tag']}/{job['dataset']}: extended={extended} "
        f"Pass@8 {before_summary.get('pass_at_k', {}).get('8', {}).get('pct', 0):.2f}% -> "
        f"{after_metrics.get('pass_at_k', {}).get('8', {}).get('pct', 0):.2f}% "
        f"len {before_summary.get('avg_output_tokens_mean', 0):.1f} -> "
        f"{after_metrics.get('avg_output_tokens_mean', 0):.1f}",
        flush=True,
    )
    return extended


def _count_left_trunc(results_doc: Dict[str, Any], *, old_max: int, target: int, trunc_slack: int) -> int:
    thr = max(0, old_max - trunc_slack)
    left = 0
    for row in results_doc.get("results") or []:
        for g in row.get("generations") or []:
            if int(g.get("extended_from") or 0) > 0:
                continue
            n = int(g.get("num_tokens") or 0)
            if n >= thr and n < target:
                left += 1
    return left


def run_shard(shard_path: Path, *, generate_batch_size: int, tp: int, gpu_mem: float, enforce_eager: bool) -> None:
    jobs = _load_jsonl(shard_path)
    if not jobs:
        print(f"[extend] empty shard: {shard_path}", flush=True)
        return

    if any(bool(j.get("allow_long_max_model_len")) for j in jobs):
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        print("[extend] VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 (pos-capped base -> 40k)", flush=True)

    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for j in jobs:
        by_model[j["model_path"]].append(j)

    print(
        f"[extend] shard={shard_path.name} jobs={len(jobs)} models={len(by_model)} "
        f"trunc={sum(j['trunc_count'] for j in jobs)}",
        flush=True,
    )

    for model_path, group in by_model.items():
        # Skip if all already extended.
        pending = []
        for j in group:
            mp = Path(j["metrics_json"])
            if not mp.is_file():
                continue
            m = json.loads(mp.read_text(encoding="utf-8"))
            if int(m.get("extended_to") or 0) >= int(j["target_max_new"]):
                # Re-check remaining truncations (skip already-continued gens).
                rp = Path(j["results_json"])
                rd = json.loads(rp.read_text(encoding="utf-8"))
                left = _count_left_trunc(
                    rd,
                    old_max=int(j["old_max_new"]),
                    target=int(j["target_max_new"]),
                    trunc_slack=int(j.get("trunc_slack") or 0),
                )
                if left == 0:
                    print(f"[extend] already done: {j['eval_tag']}/{j['dataset']}", flush=True)
                    continue
            pending.append(j)
        if not pending:
            continue

        max_model_len = max(int(j.get("max_model_len") or 40960) for j in pending)
        vllm_path = pending[0].get("vllm_base_model_path") or model_path
        allow_long = any(bool(j.get("allow_long_max_model_len")) for j in pending)
        if allow_long:
            os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        else:
            cfg_max = max_seq_len_from_model_config(vllm_path)
            if cfg_max is not None and max_model_len > cfg_max:
                print(f"[extend] capping max_model_len {max_model_len} -> {cfg_max}", flush=True)
                max_model_len = cfg_max

        lora_dir = pending[0].get("lora_adapter_dir")
        lora_path = str(lora_dir) if lora_dir else None
        gbs = batch_size_for_model(model_path, generate_batch_size)
        print(
            f"[extend] loading model={model_path} max_model_len={max_model_len} "
            f"jobs={len(pending)} generate_batch_size={gbs}",
            flush=True,
        )
        llm = build_llm(
            vllm_path if not lora_path else vllm_path,
            lora_path,
            tp,
            gpu_mem,
            max_model_len,
            enforce_eager,
            True,
            64,
        )
        tokenizer = load_eval_tokenizer(model_path)
        maybe_install_olmo_chat_template(tokenizer, model_path=vllm_path)

        try:
            for j in pending:
                extend_job(
                    j,
                    llm,
                    tokenizer,
                    generate_batch_size=gbs,
                    tp=tp,
                    gpu_mem=gpu_mem,
                )
        finally:
            _destroy_llm(llm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend truncated math-eval generations to 38912")
    parser.add_argument("--shard-file", type=str, required=True)
    parser.add_argument(
        "--generate-batch-size",
        type=int,
        default=0,
        help="0 = auto by model size (0.6b/1.7b=12, 4b=8, 8b=4)",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    args = parser.parse_args()
    run_shard(
        Path(args.shard_file).expanduser().resolve(),
        generate_batch_size=args.generate_batch_size,
        tp=args.tensor_parallel_size,
        gpu_mem=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )


if __name__ == "__main__":
    main()
