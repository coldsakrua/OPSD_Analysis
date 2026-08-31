#!/usr/bin/env python3
"""Smoke: length + math_dapo acc of an SFT ckpt on DAPO-Math (think mode)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from verl.utils.reward_score import math_dapo


def _as_messages(prompt_obj: Any) -> list[dict[str, str]]:
    if isinstance(prompt_obj, list):
        return [{"role": str(x.get("role", "user")), "content": str(x.get("content", ""))} for x in prompt_obj]
    if hasattr(prompt_obj, "tolist"):
        return _as_messages(prompt_obj.tolist())
    raise TypeError(f"unsupported prompt type: {type(prompt_obj)}")


def _ground_truth(rm: Any) -> str:
    if isinstance(rm, dict):
        return str(rm.get("ground_truth", ""))
    if hasattr(rm, "item"):
        return _ground_truth(rm.item())
    raise TypeError(f"unsupported reward_model type: {type(rm)}")


def main() -> None:
    for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        os.environ.pop(key, None)

    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--chat-template-path", default="")
    p.add_argument("--dataset", required=True)
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=12288)
    p.add_argument("--max-model-len", type=int, default=13312)  # 1024 prompt + 12288
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-json", required=True)
    args = p.parse_args()

    df = pd.read_parquet(args.dataset)
    n = min(int(args.num_samples), len(df))
    sample = df.sample(n=n, random_state=args.seed).reset_index(drop=True)
    print(f"[smoke] dataset={args.dataset} rows={len(df)} sample={n}", flush=True)

    tok_path = args.chat_template_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if args.chat_template_path and args.chat_template_path != args.model_path:
        # Keep SFT weights, overlay instruct/base chat template like GRPO.
        model_tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if getattr(tokenizer, "chat_template", None):
            model_tok.chat_template = tokenizer.chat_template
        tokenizer = model_tok

    prompts: list[str] = []
    gts: list[str] = []
    for i in range(n):
        messages = _as_messages(sample.iloc[i]["prompt"])
        gt = _ground_truth(sample.iloc[i]["reward_model"])
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(args.enable_thinking),
        )
        prompts.append(text)
        gts.append(gt)

    print(
        f"[smoke] model={args.model_path} tp={args.tensor_parallel_size} "
        f"max_new={args.max_new_tokens} think={args.enable_thinking}",
        flush=True,
    )
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    # Re-bind chat template onto the tokenizer vLLM loaded if needed.
    if getattr(tokenizer, "chat_template", None):
        try:
            llm.get_tokenizer().chat_template = tokenizer.chat_template
        except Exception:
            pass

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        n=1,
    )
    outputs = llm.generate(prompts, sp)

    rows = []
    lengths: list[int] = []
    accs: list[float] = []
    clipped = 0
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        # Prefer completion token count from vLLM when available.
        tok_len = len(out.outputs[0].token_ids) if out.outputs[0].token_ids is not None else None
        if tok_len is None:
            tok_len = len(tokenizer.encode(text, add_special_tokens=False))
        lengths.append(int(tok_len))
        if int(tok_len) >= args.max_new_tokens:
            clipped += 1
        scored = math_dapo.compute_score(text, gts[i])
        acc = float(bool(scored.get("acc")))
        accs.append(acc)
        rows.append(
            {
                "idx": i,
                "ground_truth": gts[i],
                "pred": scored.get("pred"),
                "score": scored.get("score"),
                "acc": acc,
                "response_len": int(tok_len),
                "clipped": int(tok_len) >= args.max_new_tokens,
                "prompt": prompts[i],
                "response": text,
            }
        )

    summary = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "num_samples": n,
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": bool(args.enable_thinking),
        "acc_mean": float(sum(accs) / max(1, len(accs))),
        "acc_count": int(sum(accs)),
        "resp_len_mean": float(statistics.mean(lengths)) if lengths else 0.0,
        "resp_len_median": float(statistics.median(lengths)) if lengths else 0.0,
        "resp_len_min": int(min(lengths)) if lengths else 0,
        "resp_len_max": int(max(lengths)) if lengths else 0,
        "clip_ratio": float(clipped / max(1, len(lengths))),
        "clip_count": clipped,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": rows}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[smoke] ===== SUMMARY =====", flush=True)
    for k, v in summary.items():
        print(f"[smoke] {k}: {v}", flush=True)
    print(f"[smoke] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
