#!/usr/bin/env python3
"""OPSD student/teacher on-policy analysis.

Sections 2.1–2.5 share this entrypoint. Shell wrappers set --task and model/combo args.

Pipeline
--------
1. Sample prompts from preprocessed OpenThoughts parquet (matching training collator).
2. Generate student rollouts (vLLM or SGLang).
3. Score each completion token with student + teacher (same weights, different prompts).
4. Aggregate KL/JSD, top-k KL, log-ratio, argmax preference, SNR, loss-dominant tokens.
   Persist per-token arrays to token_metrics*.jsonl when enabled.

Tasks
-----
- combinations (2.1, 2.4): single teacher prefix (opsd/solution).
- teacher_prefix (2.2): sol + answer + irrelevant_other_sol on same rollouts.
- entropy (2.3): score once, aggregate he20/le20/he80/le80 buckets from student entropy.
- length_windows (2.5): bucket metrics by completion position.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from common.aggregation import MetricsAggregator  # noqa: E402
from common.generation import run_generation  # noqa: E402
from common.io import load_jsonl, save_jsonl, write_json  # noqa: E402
from common.metrics import DEFAULT_JSD_BETA, DEFAULT_JSD_TOKEN_CLIP, DEFAULT_TOPK_HIT_KS  # noqa: E402
from common.model_registry import (  # noqa: E402
    DEFAULT_MAX_PROMPT,
    ENTROPY_BUCKETS,
    LENGTH_WINDOWS,
    TEACHER_PREFIXES,
    dataset_path,
    entropy_ratio,
    get_model_config,
    model_launch_overrides,
    task_default_gen_batch_hint,
    task_default_max_completion,
    task_default_score_batch,
    teacher_prefix_dataset,
)
from common.prompts import load_multi_prefix_samples, load_prompt_samples  # noqa: E402
from common.scoring import (  # noqa: E402
    apply_entropy_bucket,
    apply_length_window,
    rollout_metrics_summary,
    score_rollout_batch,
    token_metrics_record,
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
    p.add_argument(
        "--score-batch-size",
        type=int,
        default=0,
        help="HF score microbatch (0 = auto from task+model; short tasks 2–8, length_windows 1–2)",
    )
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
    p.add_argument(
        "--gen-batch-hint",
        type=int,
        default=0,
        help="SGLang prompt chunk (0 = auto from task+model; vLLM ignores)",
    )
    p.add_argument("--entropy-bucket", choices=("he20", "le20", "he80", "le80"), default="")
    p.add_argument("--teacher-prefixes", nargs="+", default=list(TEACHER_PREFIXES))
    p.add_argument("--top-token-k", type=int, default=50)
    p.add_argument(
        "--jsd-beta",
        type=float,
        default=DEFAULT_JSD_BETA,
        help="Mixture beta for generalized_jsd_loss (train_opsd hardcodes 0.0 = forward KL)",
    )
    p.add_argument(
        "--jsd-token-clip",
        type=float,
        default=DEFAULT_JSD_TOKEN_CLIP,
        help="Per-token JSD clip threshold (train jsd005 uses 0.05)",
    )
    p.add_argument(
        "--topk-hit-ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOPK_HIT_KS),
        help="Top-k hit diagnostics for sampled token (student/teacher)",
    )
    p.add_argument(
        "--save-token-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-token arrays (incl. SNR, argmax) to token_metrics*.jsonl",
    )
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


def _make_aggregator(args: argparse.Namespace) -> MetricsAggregator:
    return MetricsAggregator(topk_names=("k1", "k16"), topk_hit_ks=tuple(args.topk_hit_ks))


def _score_batch_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "topk_ks": (1, 16),
        "topk_hit_ks": tuple(args.topk_hit_ks),
        "jsd_beta": args.jsd_beta,
        "jsd_token_clip": args.jsd_token_clip,
    }


def score_rollouts(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    rollouts: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    teacher_key: str = "teacher_prompt",
    filter_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    window_label: str | None = None,
    token_metrics_out: list[dict[str, Any]] | None = None,
    rollout_metrics_out: list[dict[str, Any]] | None = None,
    metrics_tag: str = "",
) -> MetricsAggregator:
    agg = _make_aggregator(args)
    bs = max(1, args.score_batch_size)
    score_kw = _score_batch_kwargs(args)
    for start in tqdm(range(0, len(rollouts), bs), desc=f"score/{teacher_key}"):
        batch = rollouts[start : start + bs]
        metrics_list = score_rollout_batch(
            model,
            model,
            tokenizer,
            student_prompts=[r["student_prompt"] for r in batch],
            teacher_prompts=[r[teacher_key] for r in batch],
            completion_ids_list=[r["completion_token_ids"] for r in batch],
            **score_kw,
        )
        for r, metrics in zip(batch, metrics_list):
            if filter_fn is not None:
                metrics = filter_fn(metrics)
            agg.update(metrics, window_label=window_label)
            if rollout_metrics_out is not None:
                rollout_rec = rollout_metrics_summary(metrics)
                rollout_rec.update(
                    {
                        "row_id": r.get("row_id"),
                        "rollout_idx": r.get("rollout_idx"),
                        "teacher_key": teacher_key,
                    }
                )
                if metrics_tag:
                    rollout_rec["tag"] = metrics_tag
                if window_label:
                    rollout_rec["window"] = window_label
                rollout_metrics_out.append(rollout_rec)
            if token_metrics_out is not None and args.save_token_metrics:
                meta = {
                    "row_id": r.get("row_id"),
                    "rollout_idx": r.get("rollout_idx"),
                    "teacher_key": teacher_key,
                }
                if metrics_tag:
                    meta["tag"] = metrics_tag
                if window_label:
                    meta["window"] = window_label
                token_metrics_out.append(token_metrics_record(metrics, meta=meta))
    return agg


def run_combinations(
    args: argparse.Namespace,
    model_cfg,
    out_dir: Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
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
            ds, tokenizer, model_key=args.model_key, combo=args.combo, num_prompts=args.num_prompts,
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

    token_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    agg = score_rollouts(
        model,
        tokenizer,
        rollouts,
        args,
        filter_fn=filter_fn,
        token_metrics_out=token_rows if args.save_token_metrics else None,
        rollout_metrics_out=rollout_rows,
        metrics_tag=args.entropy_bucket or "all",
    )
    save_jsonl(out_dir / "rollout_metrics.jsonl", rollout_rows)
    if args.save_token_metrics:
        save_jsonl(out_dir / "token_metrics.jsonl", token_rows)

    summary = {
        "config": _config_dict(args, model_cfg, ds),
        "metrics": agg.summary(tokenizer, args.top_token_k),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def run_teacher_prefix(
    args: argparse.Namespace,
    model_cfg,
    out_dir: Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
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
            ds_map, tokenizer, model_key=args.model_key, combo=args.combo, num_prompts=args.num_prompts,
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
        token_rows: list[dict[str, Any]] = []
        rollout_rows: list[dict[str, Any]] = []
        agg = score_rollouts(
            model,
            tokenizer,
            rollouts,
            args,
            teacher_key=key,
            token_metrics_out=token_rows if args.save_token_metrics else None,
            rollout_metrics_out=rollout_rows,
            metrics_tag=prefix,
        )
        summaries[prefix] = agg.summary(tokenizer, args.top_token_k)
        save_jsonl(out_dir / f"rollout_metrics_{prefix}.jsonl", rollout_rows)
        if args.save_token_metrics:
            save_jsonl(out_dir / f"token_metrics_{prefix}.jsonl", token_rows)

    out = {"config": _config_dict(args, model_cfg, ds_map), "teacher_prefixes": summaries}
    write_json(out_dir / "summary.json", out)
    return out


def run_entropy(
    args: argparse.Namespace,
    model_cfg,
    out_dir: Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
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
            ds, tokenizer, model_key=args.model_key, combo=args.combo, num_prompts=args.num_prompts,
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
    bucket_summaries: dict[str, MetricsAggregator] = {}
    token_rows_full: list[dict[str, Any]] = []
    rollout_rows_full: list[dict[str, Any]] = []
    per_bucket_rows: dict[str, list[dict[str, Any]]] = {b: [] for b in ENTROPY_BUCKETS}
    per_bucket_token_rows: dict[str, list[dict[str, Any]]] = {b: [] for b in ENTROPY_BUCKETS}

    bs = max(1, args.score_batch_size)
    score_kw = _score_batch_kwargs(args)
    for start in tqdm(range(0, len(rollouts), bs), desc="score/entropy"):
        batch = rollouts[start : start + bs]
        metrics_list = score_rollout_batch(
            model,
            model,
            tokenizer,
            student_prompts=[r["student_prompt"] for r in batch],
            teacher_prompts=[r["teacher_prompt"] for r in batch],
            completion_ids_list=[r["completion_token_ids"] for r in batch],
            **score_kw,
        )
        for r, metrics in zip(batch, metrics_list):
            rollout_rec = rollout_metrics_summary(metrics)
            rollout_rec.update(
                {
                    "row_id": r.get("row_id"),
                    "rollout_idx": r.get("rollout_idx"),
                    "tag": "full",
                }
            )
            rollout_rows_full.append(rollout_rec)
            if args.save_token_metrics:
                token_rows_full.append(
                    token_metrics_record(
                        metrics,
                        meta={
                            "row_id": r.get("row_id"),
                            "rollout_idx": r.get("rollout_idx"),
                            "tag": "full",
                        },
                    )
                )
            for bucket in ENTROPY_BUCKETS:
                bucketed = apply_entropy_bucket(metrics, bucket)
                if bucket not in bucket_summaries:
                    bucket_summaries[bucket] = _make_aggregator(args)
                bucket_summaries[bucket].update(bucketed, window_label=bucket)
                if bucketed.get("n_tokens", 0) > 0:
                    bucket_rec = rollout_metrics_summary(bucketed)
                    bucket_rec.update(
                        {
                            "row_id": r.get("row_id"),
                            "rollout_idx": r.get("rollout_idx"),
                            "tag": bucket,
                        }
                    )
                    per_bucket_rows[bucket].append(bucket_rec)
                    if args.save_token_metrics:
                        per_bucket_token_rows[bucket].append(
                            token_metrics_record(
                                bucketed,
                                meta={
                                    "row_id": r.get("row_id"),
                                    "rollout_idx": r.get("rollout_idx"),
                                    "tag": bucket,
                                },
                            )
                        )

    summarized = {
        bucket: agg.summary(tokenizer, args.top_token_k) for bucket, agg in bucket_summaries.items()
    }
    save_jsonl(out_dir / "rollout_metrics.jsonl", rollout_rows_full)
    for bucket, rows in per_bucket_rows.items():
        save_jsonl(out_dir / f"rollout_metrics_{bucket}.jsonl", rows)
    if args.save_token_metrics:
        save_jsonl(out_dir / "token_metrics.jsonl", token_rows_full)
        for bucket, rows in per_bucket_token_rows.items():
            save_jsonl(out_dir / f"token_metrics_{bucket}.jsonl", rows)

    out = {"config": _config_dict(args, model_cfg, ds), "entropy_buckets": summarized}
    write_json(out_dir / "summary.json", out)
    return out


def run_length_windows(
    args: argparse.Namespace,
    model_cfg,
    out_dir: Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
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
            ds, tokenizer, model_key=args.model_key, combo=args.combo, num_prompts=args.num_prompts,
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
    window_summaries: dict[str, MetricsAggregator] = {}
    token_rows_full: list[dict[str, Any]] = []
    rollout_rows_full: list[dict[str, Any]] = []
    per_window_rows: dict[str, list[dict[str, Any]]] = {f"{s}_{e}": [] for s, e in LENGTH_WINDOWS}

    bs = max(1, args.score_batch_size)
    score_kw = _score_batch_kwargs(args)
    for start in tqdm(range(0, len(rollouts), bs), desc="score/length_windows"):
        batch = rollouts[start : start + bs]
        metrics_list = score_rollout_batch(
            model,
            model,
            tokenizer,
            student_prompts=[r["student_prompt"] for r in batch],
            teacher_prompts=[r["teacher_prompt"] for r in batch],
            completion_ids_list=[r["completion_token_ids"] for r in batch],
            **score_kw,
        )
        for r, metrics in zip(batch, metrics_list):
            rollout_rec = rollout_metrics_summary(metrics)
            rollout_rec.update(
                {
                    "row_id": r.get("row_id"),
                    "rollout_idx": r.get("rollout_idx"),
                    "tag": "full",
                }
            )
            rollout_rows_full.append(rollout_rec)
            if args.save_token_metrics:
                token_rows_full.append(
                    token_metrics_record(
                        metrics,
                        meta={
                            "row_id": r.get("row_id"),
                            "rollout_idx": r.get("rollout_idx"),
                            "tag": "full",
                        },
                    )
                )
            for win_start, win_end in LENGTH_WINDOWS:
                label = f"{win_start}_{win_end}"
                windowed = apply_length_window(metrics, win_start, win_end)
                if label not in window_summaries:
                    window_summaries[label] = _make_aggregator(args)
                window_summaries[label].update(windowed, window_label=label)
                if args.save_token_metrics and windowed.get("n_tokens", 0) > 0:
                    per_window_rows[label].append(
                        token_metrics_record(
                            windowed,
                            meta={
                                "row_id": r.get("row_id"),
                                "rollout_idx": r.get("rollout_idx"),
                                "window": label,
                            },
                        )
                    )

    summarized = {
        label: agg.summary(tokenizer, args.top_token_k) for label, agg in window_summaries.items()
    }
    if args.save_token_metrics:
        save_jsonl(out_dir / "token_metrics.jsonl", token_rows_full)
        for label, rows in per_window_rows.items():
            save_jsonl(out_dir / f"token_metrics_{label}.jsonl", rows)
    save_jsonl(out_dir / "rollout_metrics.jsonl", rollout_rows_full)

    out = {"config": _config_dict(args, model_cfg, ds), "length_windows": summarized}
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
        "score_batch_size": args.score_batch_size,
        "gen_batch_hint": args.gen_batch_hint,
        "jsd_beta": args.jsd_beta,
        "jsd_token_clip": args.jsd_token_clip,
        "topk_hit_ks": args.topk_hit_ks,
        "entropy_bucket": args.entropy_bucket,
        "entropy_kind": kind,
        "entropy_ratio": ratio,
        "teacher_prefixes": args.teacher_prefixes,
        "save_token_metrics": args.save_token_metrics,
        "metrics": {
            "jsd_kl": f"generalized_jsd_loss beta={args.jsd_beta} full vocab",
            "jsd_kl_clipped": f"min(jsd_kl, {args.jsd_token_clip}) — aligns with train jsd_token_clip",
            "frac_jsd_clipped": f"fraction of tokens with jsd_kl > {args.jsd_token_clip}",
            "topk_kl_k16": "teacher top-16 renormalized KL",
            "log_ratio_k1": "log pi_S(x) - log pi_T(x) for sampled token",
            "advantage": "log pi_T(x) - log pi_S(x); encourage/discourage counts",
            "topk_hit": f"sampled token in student/teacher TopK for K={args.topk_hit_ks}",
            "rank_within_topk_max": f"rank of sampled token in Top-{max(args.topk_hit_ks)} (capped if miss)",
            "entropy_gap": "H(pi_S) - H(pi_T) per token",
            "confidence_gap": "max log pi - log pi(x) for student and teacher",
            "snr": "|advantage| / (student_entropy + 1e-8)",
            "top1_agree": "student_argmax == teacher_argmax",
            "rollout_metrics.jsonl": "per-rollout means, frac_encourage, first/last-128 advantage, clip rate",
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
    if args.score_batch_size <= 0:
        args.score_batch_size = task_default_score_batch(args.task, args.model_key)
    if args.gen_batch_hint <= 0:
        args.gen_batch_hint = task_default_gen_batch_hint(args.task, args.model_key)
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
    print(f"[cfg] max_completion={args.max_completion_length} backend={args.backend} jsd_beta={args.jsd_beta}", flush=True)
    print(
        f"[cfg] score_batch={args.score_batch_size} gen_batch_hint={args.gen_batch_hint}",
        flush=True,
    )
    print(f"[cfg] output={out_dir}", flush=True)

    if args.task == "combinations":
        result = run_combinations(args, model_cfg, out_dir, overrides)
    elif args.task == "entropy":
        result = run_entropy(args, model_cfg, out_dir, overrides)
    elif args.task == "teacher_prefix":
        result = run_teacher_prefix(args, model_cfg, out_dir, overrides)
    elif args.task == "length_windows":
        result = run_length_windows(args, model_cfg, out_dir, overrides)
    else:
        raise ValueError(args.task)

    print(f"[done] {time.time() - t0:.1f}s", flush=True)
    if isinstance(result, dict) and "metrics" in result:
        m = result["metrics"]
        print(
            f"  mean_jsd_kl={m.get('mean_jsd_kl', 0):.4f} "
            f"top1_agree={m.get('top1_agree_rate', 0):.3f} "
            f"mean_snr={m.get('mean_snr', 0):.4f} "
            f"n_tokens={m.get('n_tokens', 0)}",
            flush=True,
        )
    elif isinstance(result, dict) and "teacher_prefixes" in result:
        for name, m in result["teacher_prefixes"].items():
            print(
                f"  [{name}] mean_jsd_kl={m.get('mean_jsd_kl', 0):.4f} "
                f"top1_agree={m.get('top1_agree_rate', 0):.3f} "
                f"mean_snr={m.get('mean_snr', 0):.4f}",
                flush=True,
            )
    elif isinstance(result, dict) and "length_windows" in result:
        for name, m in result["length_windows"].items():
            print(
                f"  [{name}] mean_jsd_kl={m.get('mean_jsd_kl', 0):.4f} "
                f"top1_agree={m.get('top1_agree_rate', 0):.3f} "
                f"mean_snr={m.get('mean_snr', 0):.4f} n={m.get('n_tokens', 0)}",
                flush=True,
            )
    elif isinstance(result, dict) and "entropy_buckets" in result:
        for name, m in result["entropy_buckets"].items():
            print(
                f"  [{name}] mean_jsd_kl={m.get('mean_jsd_kl', 0):.4f} "
                f"top1_agree={m.get('top1_agree_rate', 0):.3f} "
                f"mean_snr={m.get('mean_snr', 0):.4f} n={m.get('n_tokens', 0)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
