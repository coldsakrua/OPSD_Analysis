"""Student rollout generation (vLLM / SGLang)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def _rollout_rec(sample: dict[str, Any], rollout_idx: int, tok_ids: list[int], text: str, finish_reason: Any) -> dict[str, Any]:
    rec = {k: v for k, v in sample.items() if not k.startswith("teacher_prompt_len")}
    rec.update(
        {
            "rollout_idx": rollout_idx,
            "completion_token_ids": tok_ids,
            "completion_len": len(tok_ids),
            "completion_text": text,
            "finish_reason": finish_reason,
        }
    )
    return rec


def run_generation_vllm(
    model_path: str,
    samples: list[dict[str, Any]],
    *,
    n_rollouts: int,
    max_prompt_length: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    gpu_memory_utilization: float,
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        # Match eval: prompt + completion (no pad). Qwen think: 2048+38912=40960.
        max_model_len=max_prompt_length + max_completion_length,
        disable_custom_all_reduce=True,
    )
    sampling = SamplingParams(
        n=n_rollouts,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_completion_length,
        seed=seed,
    )
    prompts = [s["student_prompt"] for s in samples]
    print(f"[gen] vllm prompts={len(prompts)} n={n_rollouts} max_tokens={max_completion_length}", flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    rollouts: list[dict[str, Any]] = []
    lengths: list[int] = []
    for sample, out in zip(samples, outputs):
        for ri, cand in enumerate(out.outputs):
            tok_ids = list(cand.token_ids)
            lengths.append(len(tok_ids))
            rollouts.append(_rollout_rec(sample, ri, tok_ids, cand.text, getattr(cand, "finish_reason", None)))
    arr = np.asarray(lengths, dtype=np.float64)
    print(
        f"[gen] done {time.time() - t0:.1f}s n={len(rollouts)} mean_len={arr.mean():.1f}",
        flush=True,
    )
    del llm
    torch.cuda.empty_cache()
    return rollouts


def run_generation_sglang(
    model_path: str,
    samples: list[dict[str, Any]],
    *,
    n_rollouts: int,
    max_prompt_length: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    gen_batch_hint: int = 64,
    attention_backend: str = "triton",
    sampling_backend: str = "pytorch",
    mem_fraction_static: float = 0.80,
    reasoning_parser: str = "",
    disable_piecewise_cuda_graph: bool = False,
) -> list[dict[str, Any]]:
    from sglang import Engine

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    context_length = max_prompt_length + max_completion_length
    engine_kwargs: dict[str, Any] = {
        "model_path": model_path,
        "tokenizer_path": model_path,
        "trust_remote_code": True,
        "attention_backend": attention_backend,
        "sampling_backend": sampling_backend,
        "mem_fraction_static": mem_fraction_static,
        "tp_size": 1,
        "context_length": context_length,
        "log_level": "error",
    }
    if reasoning_parser:
        engine_kwargs["reasoning_parser"] = reasoning_parser
    if disable_piecewise_cuda_graph:
        engine_kwargs["disable_piecewise_cuda_graph"] = True
    llm = Engine(**engine_kwargs)

    sampling_params: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_completion_length,
        "n": n_rollouts,
        "sampling_seed": seed,
        "skip_special_tokens": True,
    }
    if top_k > 0:
        sampling_params["top_k"] = top_k

    prompts = [s["student_prompt"] for s in samples]
    gbs = max(1, gen_batch_hint)
    rollouts: list[dict[str, Any]] = []
    lengths: list[int] = []
    t0 = time.time()
    try:
        for start in tqdm(range(0, len(prompts), gbs), desc="sglang_gen"):
            chunk_prompts = prompts[start : start + gbs]
            chunk_samples = samples[start : start + gbs]
            raw = llm.generate(chunk_prompts, sampling_params=sampling_params)
            if isinstance(raw, dict):
                raw = [raw]
            gen_n = n_rollouts
            if len(raw) == len(chunk_prompts) * gen_n:
                grouped = [raw[i * gen_n : (i + 1) * gen_n] for i in range(len(chunk_prompts))]
            elif len(raw) == len(chunk_prompts):
                grouped = []
                for item in raw:
                    texts = item.get("text")
                    if isinstance(texts, list):
                        grouped.append([{"text": t, "meta_info": item.get("meta_info") or {}} for t in texts[:gen_n]])
                    else:
                        grouped.append([item])
            else:
                raise RuntimeError(f"unexpected SGLang output len={len(raw)}")
            for sample, outs in zip(chunk_samples, grouped):
                for ri, item in enumerate(outs):
                    ids = item.get("output_ids")
                    if isinstance(ids, list) and ids:
                        tok_ids = [int(x) for x in ids]
                    else:
                        text = item.get("text") or ""
                        if isinstance(text, list):
                            text = text[0] if text else ""
                        tok_ids = tokenizer.encode(str(text), add_special_tokens=False)
                    text = item.get("text") or ""
                    if isinstance(text, list):
                        text = text[0] if text else ""
                    lengths.append(len(tok_ids))
                    meta = item.get("meta_info") or {}
                    rollouts.append(_rollout_rec(sample, ri, tok_ids, str(text), meta.get("finish_reason")))
    finally:
        llm.shutdown()
        torch.cuda.empty_cache()
    arr = np.asarray(lengths, dtype=np.float64)
    print(f"[gen] sglang done {time.time() - t0:.1f}s n={len(rollouts)} mean_len={arr.mean():.1f}", flush=True)
    return rollouts


def run_generation(
    model_path: str,
    samples: list[dict[str, Any]],
    *,
    backend: str,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    common = (
        "n_rollouts",
        "max_prompt_length",
        "max_completion_length",
        "temperature",
        "top_p",
        "top_k",
        "seed",
    )
    base = {k: kwargs[k] for k in common if k in kwargs}
    if backend == "sglang":
        sglang_defaults: dict[str, Any] = {
            "gen_batch_hint": 64,
            "attention_backend": "triton",
            "sampling_backend": "pytorch",
            "mem_fraction_static": 0.80,
            "reasoning_parser": "",
            "disable_piecewise_cuda_graph": False,
        }
        for k, default in sglang_defaults.items():
            val = kwargs.get(k, default)
            base[k] = default if val is None else val
        return run_generation_sglang(model_path, samples, **base)
    if "gpu_memory_utilization" in kwargs:
        base["gpu_memory_utilization"] = kwargs["gpu_memory_utilization"]
    return run_generation_vllm(model_path, samples, **base)
