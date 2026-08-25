#!/usr/bin/env python3
"""OPSD student/teacher on-policy analysis.

Sections 2.1–2.5 share this entrypoint. Shell wrappers set --task and model/combo args.

Pipeline
--------
1. Sample prompts from preprocessed OpenThoughts parquet (matching training collator).
2. Generate student rollouts (vLLM or SGLang).
3. Score each completion token with student + teacher (same weights, different prompts).
4. Aggregate JSD KL, top-k KL (k=1,16), log-ratio, loss-dominant tokens.

Tasks
-----
- combinations (2.1, 2.4): single teacher prefix (opsd/solution).
- teacher_prefix (2.2): sol + answer + irrelevant_other_sol on same rollouts.
- entropy (2.3): filter positions by student entropy bucket then score.
- length_windows (2.5): bucket metrics by completion position.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from common.aggregation import MetricsAggregator, merge_summaries  # noqa: E402
from common.generation import run_generation  # noqa: E402
from common.io import load_jsonl, save_jsonl, write_json  # noqa: E402
from common.model_registry import (  # noqa: E402
    DEFAULT_MAX_PROMPT,
    LENGTH_WINDOWS,
    TEACHER_PREFIXES,
    dataset_path,
    entropy_ratio,
    get_model_config,
    model_launch_overrides,
    task_default_max_completion,
    teacher_prefix_dataset,
)
from common.prompts import load_multi_prefix_samples, load_prompt_samples  # noqa: E402
from common.scoring import (  # noqa: E402
    apply_entropy_bucket,
    apply_length_window,
    score_rollout_pair,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=("combinations", "teacher_prefix", "entropy", "length_windows"), required=True)
    p.add_argument("--model-key", required=True, help="Registry key, e.g. qwen3_1.7b")
    p.add_argument("--combo", required=True, help="st_tt | snt_tnt | st_tnt | snt_tt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", default="", help="Override registry model path")
    p.add_argument("--dataset-path", default="", help="Override default parquet path")
    p.add_argument("--num-prompts", type=int, default=2048)
    p.add_argument("--n-rollouts", type=int, default=2)
    p.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT)
    p.add_argument(
        "--max-completion-length",
        type=int,
        default=None,
        help="Default: 6144 for length_windows, 1024 otherwise",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--score-batch-size", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--backend", choices=("vllm", "sglang"), default="")
    p.add_argument("--attention-backend", default="triton")
    p.add_argument("--sampling-backend", default="pytorch")
    p.add_argument("--mem-fraction-static", type=float, default=None)
    p.add_argument("--reasoning-parser", default="")
    p.add_argument(
        "--disable-piecewise-cuda-graph",
        action="store_true",
        help="SGLang: disable piecewise CUDA graph (required for Falcon-H1R Mamba)",
    )
    p.add_argument("--gen-batch-hint", type=int, default=64)
    p.add_argument("--entropy-bucket", choices=("he20", "le20", "he80", "le80"), default="")
    p.add_argument("--teacher-prefixes", nargs="+", default=list(TEACHER_PREFIXES))
    p.add_argument("--top-token-k", type=int, default=50)
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-score", action="store_true")
    p.add_argument("--max-rollouts", type=int, default=0, help="0 = all")
    return p.parse_args()


def load_models(model_path: str) -> tuple[Any, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[score] loading HF model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def score_rollouts(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    rollouts: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    teacher_key: str = "teacher_prompt",
    filter_fn=None,
    window_label: str | None = None,
) -> MetricsAggregator:
    agg = MetricsAggregator(topk_names=("k1", "k16"))
    bs = max(1, args.score_batch_size)
    for start in tqdm(range(0, len(rollouts), bs), desc=f"score/{teacher_key}"):
        batch = rollouts[start : start + bs]
        for r in batch:
            metrics = score_rollout_pair(
                model,
                model,
                tokenizer,
                student_prompt=r["student_prompt"],
                teacher_prompt=r[teacher_key],
                completion_ids=r["completion_token_ids"],
                temperature=args.temperature,
                topk_ks=(1, 16),
            )
            if filter_fn is not None:
                metrics = filter_fn(metrics)
            agg.update(metrics, window_label=window_label)
    return agg


def run_combinations(args: argparse.Namespace, model_cfg, out_dir: Path) -> dict[str, Any]:
    ds = args.dataset_path or str(dataset_path(args.model_key, args.combo))
    rollouts_path = out_dir / "rollouts.jsonl"
    model_path = args.model_path or model_cfg.model_path

    if args.skip_generate and rollouts_path.is_file():
        rollouts = load_jsonl(rollouts_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        samples = load_prompt_samples(
            ds, tokenizer, combo=args.combo, num_prompts=args.num_prompts,
            max_prompt_length=args.max_prompt_length, seed=args.seed,
        )
        save_jsonl(out_dir / "samples.jsonl", samples)
        backend = args.backend or model_cfg.backend
        rollouts = run_generation(
            model_path,
            samples,
            backend=backend,
            **_generation_kwargs(args, overrides),
        )
        save_jsonl(rollouts_path, rollouts)

    if args.max_rollouts > 0:
        rollouts = rollouts[: args.max_rollouts]
    if args.skip_score:
        return {"status": "generate_only", "n_rollouts": len(rollouts)}

    tokenizer, model = load_models(model_path)

    filter_fn = None
    if args.entropy_bucket:
        filter_fn = lambda m: apply_entropy_bucket(m, args.entropy_bucket)

    agg = score_rollouts(model, tokenizer, rollouts, args, filter_fn=filter_fn)
    summary = {
        "config": _config_dict(args, model_cfg, ds),
        "metrics": agg.summary(tokenizer, args.top_token_k),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def run_teacher_prefix(args: argparse.Namespace, model_cfg, out_dir: Path) -> dict[str, Any]:
    ds_map = {
        name: str(teacher_prefix_dataset(args.model_key, args.combo, name))
        for name in args.teacher_prefixes
    }
    rollouts_path = out_dir / "rollouts.jsonl"
    model_path = args.model_path or model_cfg.model_path

    if args.skip_generate and rollouts_path.is_file():
        rollouts = load_jsonl(rollouts_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        samples = load_multi_prefix_samples(
            ds_map, tokenizer, combo=args.combo, num_prompts=args.num_prompts,
            max_prompt_length=args.max_prompt_length, seed=args.seed,
        )
        save_jsonl(out_dir / "samples.jsonl", samples)
        backend = args.backend or model_cfg.backend
        rollouts = run_generation(
            model_path,
            samples,
            backend=backend,
            **_generation_kwargs(args, overrides),
        )
        save_jsonl(rollouts_path, rollouts)

    if args.max_rollouts > 0:
        rollouts = rollouts[: args.max_rollouts]
    if args.skip_score:
        return {"status": "generate_only"}

    tokenizer, model = load_models(model_path)
    summaries: dict[str, Any] = {}
    for prefix in args.teacher_prefixes:
        key = f"teacher_prompt_{prefix}"
        agg = score_rollouts(model, tokenizer, rollouts, args, teacher_key=key)
        summaries[prefix] = agg.summary(tokenizer, args.top_token_k)

    out = {"config": _config_dict(args, model_cfg, ds_map), "teacher_prefixes": summaries}
    write_json(out_dir / "summary.json", out)
    return out


def run_entropy(args: argparse.Namespace, model_cfg, out_dir: Path) -> dict[str, Any]:
    if not args.entropy_bucket:
        raise ValueError("--entropy-bucket required for task=entropy")
    return run_combinations(args, model_cfg, out_dir)


def run_length_windows(args: argparse.Namespace, model_cfg, out_dir: Path) -> dict[str, Any]:
    ds = args.dataset_path or str(dataset_path(args.model_key, args.combo))
    rollouts_path = out_dir / "rollouts.jsonl"
    model_path = args.model_path or model_cfg.model_path

    if args.skip_generate and rollouts_path.is_file():
        rollouts = load_jsonl(rollouts_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        samples = load_prompt_samples(
            ds, tokenizer, combo=args.combo, num_prompts=args.num_prompts,
            max_prompt_length=args.max_prompt_length, seed=args.seed,
        )
        save_jsonl(out_dir / "samples.jsonl", samples)
        backend = args.backend or model_cfg.backend
        rollouts = run_generation(
            model_path,
            samples,
            backend=backend,
            **_generation_kwargs(args, overrides),
        )
        save_jsonl(rollouts_path, rollouts)

    if args.max_rollouts > 0:
        rollouts = rollouts[: args.max_rollouts]
    if args.skip_score:
        return {"status": "generate_only"}

    tokenizer, model = load_models(model_path)
    window_summaries: dict[str, Any] = {}
    for start, end in LENGTH_WINDOWS:
        label = f"{start}_{end}"
        agg = MetricsAggregator(topk_names=("k1", "k16"))
        for r in tqdm(rollouts, desc=f"window {label}"):
            metrics = score_rollout_pair(
                model, model, tokenizer,
                student_prompt=r["student_prompt"],
                teacher_prompt=r["teacher_prompt"],
                completion_ids=r["completion_token_ids"],
                temperature=args.temperature,
                topk_ks=(1, 16),
            )
            metrics = apply_length_window(metrics, start, end)
            agg.update(metrics, window_label=label)
        window_summaries[label] = agg.summary(tokenizer, args.top_token_k)

    out = {"config": _config_dict(args, model_cfg, ds), "length_windows": window_summaries}
    write_json(out_dir / "summary.json", out)
    return out


def _config_dict(args: argparse.Namespace, model_cfg, dataset) -> dict[str, Any]:
    kind, ratio = (entropy_ratio(args.entropy_bucket) if args.entropy_bucket else (None, None))
    return {
        "task": args.task,
        "model_key": args.model_key,
        "combo": args.combo,
        "model_path": args.model_path or model_cfg.model_path,
        "dataset": dataset,
        "num_prompts": args.num_prompts,
        "n_rollouts": args.n_rollouts,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "entropy_bucket": args.entropy_bucket,
        "entropy_kind": kind,
        "entropy_ratio": ratio,
        "teacher_prefixes": args.teacher_prefixes,
        "metrics": {
            "jsd_kl": "generalized JSD beta=0.5 full vocab",
            "topk_kl_k16": "teacher top-16 renormalized KL",
            "log_ratio_k1": "log pi_S(x) - log pi_T(x) for sampled token",
        },
    }


def _generation_kwargs(args: argparse.Namespace, overrides: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_rollouts": args.n_rollouts,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "gen_batch_hint": args.gen_batch_hint,
        "attention_backend": args.attention_backend,
        "sampling_backend": args.sampling_backend,
        "mem_fraction_static": args.mem_fraction_static,
        "reasoning_parser": args.reasoning_parser or None,
        "disable_piecewise_cuda_graph": args.disable_piecewise_cuda_graph,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = get_model_config(args.model_key)
    overrides = model_launch_overrides(args.model_key)
    if args.max_completion_length is None:
        args.max_completion_length = task_default_max_completion(args.task)
    if not args.backend:
        args.backend = overrides.get("backend", model_cfg.backend)
    if not args.reasoning_parser and model_cfg.reasoning_parser:
        args.reasoning_parser = model_cfg.reasoning_parser
    if args.mem_fraction_static is None:
        args.mem_fraction_static = overrides.get("mem_fraction_static", 0.80)
    if overrides.get("disable_piecewise_cuda_graph"):
        args.disable_piecewise_cuda_graph = True

    t0 = time.time()
    print(f"[cfg] task={args.task} model={args.model_key} combo={args.combo}", flush=True)
    print(f"[cfg] max_completion={args.max_completion_length} backend={args.backend}", flush=True)
    print(f"[cfg] output={out_dir}", flush=True)

    if args.task in ("combinations", "entropy"):
        result = run_combinations(args, model_cfg, out_dir)
    elif args.task == "teacher_prefix":
        result = run_teacher_prefix(args, model_cfg, out_dir)
    elif args.task == "length_windows":
        result = run_length_windows(args, model_cfg, out_dir)
    else:
        raise ValueError(args.task)

    print(f"[done] {time.time() - t0:.1f}s", flush=True)
    if isinstance(result, dict) and "metrics" in result:
        m = result["metrics"]
        print(f"  mean_jsd_kl={m.get('mean_jsd_kl', 0):.4f} n_tokens={m.get('n_tokens', 0)}", flush=True)


if __name__ == "__main__":
    main()
