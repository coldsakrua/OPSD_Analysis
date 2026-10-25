#!/usr/bin/env python3
"""Smoke: vLLM-generate on APPS, then extract+judge model outputs.

Uses the same extract/reward path as GRPO (``extract_lcb_candidate`` +
``reward_code_apps.compute_score``).

Example::

    python scripts/train_code/sft_rl/smoke_apps_extract.py \\
      --model-path outputs/qwen3_4b_base/sft_think_4gpu_12k_ocr/3925962/checkpoint-15000 \\
      --num-samples 4 --tensor-parallel-size 1
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval_code"))

import pyarrow.parquet as pq  # noqa: E402
from code_extract import extract_lcb_candidate  # noqa: E402
from reward_code_apps import compute_score  # noqa: E402
from stop_tokens import resolve_stop_token_ids  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def _as_messages(prompt_obj: Any) -> list[dict[str, str]]:
    if isinstance(prompt_obj, list):
        return [
            {"role": str(x.get("role", "user")), "content": str(x.get("content", ""))}
            for x in prompt_obj
        ]
    if hasattr(prompt_obj, "tolist"):
        return _as_messages(prompt_obj.tolist())
    raise TypeError(f"unsupported prompt type: {type(prompt_obj)}")


def _ground_truth(rm: Any) -> str:
    if isinstance(rm, dict):
        return str(rm.get("ground_truth", ""))
    if hasattr(rm, "item"):
        return _ground_truth(rm.item())
    raise TypeError(f"unsupported reward_model type: {type(rm)}")


def _config_max_pos(model_path: str) -> int | None:
    cfg = Path(model_path) / "config.json"
    if not cfg.is_file():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        v = data.get("max_position_embeddings")
        return int(v) if v is not None else None
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--chat-template-path", default="")
    p.add_argument(
        "--dataset",
        type=Path,
        default=ROOT
        / "data/apps/preprocessed/apps.traintest.qwen3.think.maxprompt2048.maxtests10.parquet",
    )
    p.add_argument("--num-samples", type=int, default=4, help="Number of problems")
    p.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="Samples per problem (vLLM SamplingParams.n)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--offset", type=int, default=0, help="If >0, take contiguous slice instead of sample")
    p.add_argument("--max-new-tokens", type=int, default=24576)
    p.add_argument("--max-model-len", type=int, default=26624)  # 2048 prompt + 24576
    p.add_argument("--allow-long-max-model-len", action="store_true")
    p.add_argument("--temperature", type=float, default=0.6)  # match scripts/sft_rl/common.sh
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--timeout", type=int, default=6)
    p.add_argument("--quiet-rows", action="store_true", help="Only print per-problem aggregates")
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        os.environ.pop(key, None)

    args = parse_args()
    if not Path(args.model_path).is_dir():
        raise SystemExit(f"missing model: {args.model_path}")
    if not args.dataset.is_file():
        raise SystemExit(f"missing dataset: {args.dataset}")

    table = pq.read_table(args.dataset)
    all_rows = table.to_pylist()
    if args.offset > 0:
        rows = all_rows[args.offset : args.offset + args.num_samples]
    else:
        import random

        rng = random.Random(args.seed)
        idxs = list(range(len(all_rows)))
        rng.shuffle(idxs)
        rows = [all_rows[i] for i in idxs[: args.num_samples]]
    n = len(rows)
    if n == 0:
        raise SystemExit("no rows selected")
    print(f"[smoke] dataset={args.dataset} pool={len(all_rows)} sample={n}", flush=True)

    tok_path = args.chat_template_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if args.chat_template_path and args.chat_template_path != args.model_path:
        model_tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if getattr(tokenizer, "chat_template", None):
            model_tok.chat_template = tokenizer.chat_template
        tokenizer = model_tok

    prompts: list[str] = []
    metas: list[dict[str, Any]] = []
    for row in rows:
        messages = _as_messages(row["prompt"])
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(args.enable_thinking),
        )
        prompts.append(text)
        metas.append(
            {
                "problem_id": row.get("extra_info", {}).get("problem_id"),
                "difficulty": row.get("extra_info", {}).get("difficulty"),
                "fn_name": row.get("extra_info", {}).get("fn_name"),
                "n_tests": row.get("extra_info", {}).get("n_tests"),
                "ground_truth": _ground_truth(row["reward_model"]),
                "extra_info": row.get("extra_info") or {},
            }
        )

    max_model_len = int(args.max_model_len)
    cfg_max = _config_max_pos(args.model_path)
    if cfg_max is not None and max_model_len > cfg_max and not args.allow_long_max_model_len:
        print(
            f"[smoke] capping max_model_len {max_model_len} -> {cfg_max}",
            flush=True,
        )
        max_model_len = cfg_max

    stop_ids = resolve_stop_token_ids(tokenizer)
    print(
        f"[smoke] model={args.model_path} tp={args.tensor_parallel_size} "
        f"max_new={args.max_new_tokens} max_model_len={max_model_len} "
        f"think={args.enable_thinking} stop_token_ids={stop_ids}",
        flush=True,
    )

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    if getattr(tokenizer, "chat_template", None):
        try:
            llm.get_tokenizer().chat_template = tokenizer.chat_template
        except Exception:
            pass

    n_gen = max(1, int(args.num_generations))
    sp_kw: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_new_tokens,
        "n": n_gen,
    }
    if stop_ids:
        sp_kw["stop_token_ids"] = stop_ids
    print(f"[smoke] generating problems={n} n_gen={n_gen} total={n * n_gen}", flush=True)
    outputs = llm.generate(prompts, SamplingParams(**sp_kw))

    out_rows: list[dict[str, Any]] = []
    lengths: list[int] = []
    n_extracted = 0
    n_acc = 0
    n_total = 0
    pass_at_1 = 0  # mean over gens (same as acc_mean)
    pass_at_k = 0  # any of n_gen correct
    extract_at_k = 0  # any gen extracted
    print("-" * 72, flush=True)
    for i, out in enumerate(outputs):
        problem_accs: list[bool] = []
        problem_exts: list[bool] = []
        for gi, cand in enumerate(out.outputs):
            n_total += 1
            text = cand.text or ""
            tok_len = (
                len(cand.token_ids)
                if cand.token_ids is not None
                else len(tokenizer.encode(text, add_special_tokens=False))
            )
            lengths.append(int(tok_len))
            extracted = extract_lcb_candidate(text, profile="strict")
            has_code = bool((extracted or "").strip())
            if has_code:
                n_extracted += 1
            scored = compute_score(
                data_source="apps",
                solution_str=text,
                ground_truth=metas[i]["ground_truth"],
                extra_info={**metas[i]["extra_info"], "timeout": args.timeout},
                timeout=args.timeout,
            )
            acc = bool(scored.get("acc"))
            if acc:
                n_acc += 1
            problem_accs.append(acc)
            problem_exts.append(has_code)

            if not args.quiet_rows:
                preview = text.replace("\n", "\\n")
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                print(
                    f"[smoke] id={metas[i]['problem_id']} g={gi} len={tok_len} "
                    f"extract={'Y' if has_code else 'N'}({len(extracted or '')}) "
                    f"acc={acc} score={scored.get('score')} err={scored.get('error')}",
                    flush=True,
                )
                print(f"  resp: {preview}", flush=True)

            out_rows.append(
                {
                    "idx": i,
                    "gen": gi,
                    "problem_id": metas[i]["problem_id"],
                    "difficulty": metas[i]["difficulty"],
                    "fn_name": metas[i]["fn_name"],
                    "n_tests": metas[i]["n_tests"],
                    "response_len": int(tok_len),
                    "clipped": int(tok_len) >= args.max_new_tokens,
                    "extract_len": len(extracted or ""),
                    "has_python_fence": has_code,
                    "score": scored.get("score"),
                    "acc": acc,
                    "pass_rate": scored.get("pass_rate"),
                    "error": scored.get("error"),
                    "extract": scored.get("extract"),
                    "extracted_code": extracted,
                    "response": text,
                }
            )

        any_acc = any(problem_accs)
        any_ext = any(problem_exts)
        if any_acc:
            pass_at_k += 1
        if any_ext:
            extract_at_k += 1
        print(
            f"[smoke] id={metas[i]['problem_id']} "
            f"pass@{n_gen}={int(any_acc)} extract_any={int(any_ext)} "
            f"accs={sum(problem_accs)}/{n_gen} "
            f"lens={[r['response_len'] for r in out_rows[-n_gen:]]}",
            flush=True,
        )

    summary = {
        "model_path": args.model_path,
        "dataset": str(args.dataset),
        "num_problems": n,
        "num_generations": n_gen,
        "num_total": n_total,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": max_model_len,
        "enable_thinking": bool(args.enable_thinking),
        "temperature": args.temperature,
        "extract_rate": float(n_extracted / max(1, n_total)),
        "extract_count": n_extracted,
        "extract_any_rate": float(extract_at_k / max(1, n)),
        "extract_any_count": extract_at_k,
        "acc_mean": float(n_acc / max(1, n_total)),
        "acc_count": n_acc,
        f"pass_at_{n_gen}": float(pass_at_k / max(1, n)),
        f"pass_at_{n_gen}_count": pass_at_k,
        "resp_len_mean": float(statistics.mean(lengths)) if lengths else 0.0,
        "resp_len_median": float(statistics.median(lengths)) if lengths else 0.0,
        "resp_len_max": int(max(lengths)) if lengths else 0,
        "clip_count": sum(1 for r in out_rows if r["clipped"]),
        "clip_rate": float(sum(1 for r in out_rows if r["clipped"]) / max(1, n_total)),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"summary": summary, "rows": out_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("-" * 72, flush=True)
    print("[smoke] ===== SUMMARY =====", flush=True)
    for k, v in summary.items():
        print(f"[smoke] {k}: {v}", flush=True)
    print(f"[smoke] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
