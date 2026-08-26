"""Forward-pass scoring and metric extraction."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM

from .metrics import (
    DEFAULT_JSD_BETA,
    DEFAULT_JSD_TOKEN_CLIP,
    DEFAULT_TOPK_HIT_KS,
    argmax_token_ids,
    clip_jsd_per_token,
    compute_jsd_kl,
    compute_snr,
    compute_token_entropy,
    compute_topk_hit_and_ranks,
    compute_topk_kl,
    confidence_gap,
    entropy_mask,
    log_ratio_k1,
    sampled_logprobs,
)

PER_TOKEN_SCALAR_KEYS = (
    "jsd_kl",
    "jsd_kl_clipped",
    "jsd_would_clip",
    "log_ratio_k1",
    "advantage",
    "student_entropy",
    "teacher_entropy",
    "entropy_gap",
    "snr",
    "top1_agree",
    "student_logp",
    "teacher_logp",
    "student_confidence_gap",
    "teacher_confidence_gap",
    "rank_student_topk",
    "rank_teacher_topk",
)
PER_TOKEN_ID_KEYS = (
    "token_ids",
    "student_argmax_ids",
    "teacher_argmax_ids",
)


@torch.no_grad()
def forward_completion_logits(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prompt_texts: list[str],
    completion_ids_list: list[list[int]],
) -> list[torch.Tensor]:
    """Return per-example completion logits [L, V] on CPU (float32)."""
    if not prompt_texts:
        return []
    prompt_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompt_texts]
    seqs = [p + c for p, c in zip(prompt_ids, completion_ids_list)]
    plens = [len(p) for p in prompt_ids]
    clens = [len(c) for c in completion_ids_list]
    max_len = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    batch = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=model.device)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=model.device)
    for i, s in enumerate(seqs):
        batch[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=model.device)
        attn[i, : len(s)] = 1

    out = model(input_ids=batch, attention_mask=attn)
    logits = out.logits.float()

    results: list[torch.Tensor] = []
    for i, (plen, clen) in enumerate(zip(plens, clens)):
        if clen == 0:
            results.append(torch.empty(0, 0))
            continue
        pos_logits = logits[i, plen - 1 : plen - 1 + clen, :].cpu()
        results.append(pos_logits)
    return results


def _metrics_from_logits(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    completion_ids: list[int],
    *,
    temperature: float,
    topk_ks: tuple[int, ...],
    topk_hit_ks: tuple[int, ...],
    jsd_beta: float,
    jsd_token_clip: float,
) -> dict[str, Any]:
    """Per-position metrics from completion logits [L, V] on device."""
    L = min(s_logits.shape[0], t_logits.shape[0], len(completion_ids))
    if L == 0:
        return {"n_tokens": 0}

    s_logits = s_logits[:L]
    t_logits = t_logits[:L]
    tok = torch.tensor(completion_ids[:L], dtype=torch.long, device=s_logits.device)

    jsd = compute_jsd_kl(s_logits, t_logits, beta=jsd_beta, temperature=temperature).cpu()
    jsd_clipped, jsd_would_clip = clip_jsd_per_token(
        jsd.to(s_logits.device), jsd_token_clip
    )
    jsd_clipped = jsd_clipped.cpu()
    jsd_would_clip = jsd_would_clip.cpu()

    lr1 = log_ratio_k1(s_logits, t_logits, tok, temperature=temperature).cpu()
    advantage = (-lr1).contiguous()
    s_ent = compute_token_entropy(s_logits, temperature=temperature).cpu()
    t_ent = compute_token_entropy(t_logits, temperature=temperature).cpu()
    ent_gap = (s_ent - t_ent).contiguous()
    snr = compute_snr(advantage, s_ent).cpu()
    s_arg = argmax_token_ids(s_logits, temperature=temperature).cpu()
    t_arg = argmax_token_ids(t_logits, temperature=temperature).cpu()
    top1_agree = (s_arg == t_arg).to(torch.float32)

    s_logp = sampled_logprobs(s_logits, tok, temperature=temperature).cpu()
    t_logp = sampled_logprobs(t_logits, tok, temperature=temperature).cpu()
    s_conf_gap = confidence_gap(s_logits, tok, temperature=temperature).cpu()
    t_conf_gap = confidence_gap(t_logits, tok, temperature=temperature).cpu()

    s_hits, s_rank = compute_topk_hit_and_ranks(
        s_logits, tok, temperature=temperature, ks=topk_hit_ks
    )
    t_hits, t_rank = compute_topk_hit_and_ranks(
        t_logits, tok, temperature=temperature, ks=topk_hit_ks
    )
    topk_hit = {
        str(k): {
            "student": s_hits[k].cpu().tolist(),
            "teacher": t_hits[k].cpu().tolist(),
        }
        for k in topk_hit_ks
    }

    topk: dict[str, torch.Tensor] = {}
    for k in topk_ks:
        if k == 1:
            topk["k1"] = lr1
        else:
            topk[f"k{k}"] = compute_topk_kl(s_logits, t_logits, k=k, temperature=temperature).cpu()

    return {
        "n_tokens": L,
        "token_ids": completion_ids[:L],
        "jsd_kl": jsd.numpy().tolist(),
        "jsd_kl_clipped": jsd_clipped.numpy().tolist(),
        "jsd_would_clip": jsd_would_clip.numpy().tolist(),
        "log_ratio_k1": lr1.numpy().tolist(),
        "advantage": advantage.numpy().tolist(),
        "student_entropy": s_ent.numpy().tolist(),
        "teacher_entropy": t_ent.numpy().tolist(),
        "entropy_gap": ent_gap.numpy().tolist(),
        "snr": snr.numpy().tolist(),
        "student_logp": s_logp.numpy().tolist(),
        "teacher_logp": t_logp.numpy().tolist(),
        "student_confidence_gap": s_conf_gap.numpy().tolist(),
        "teacher_confidence_gap": t_conf_gap.numpy().tolist(),
        "rank_student_topk": s_rank.cpu().tolist(),
        "rank_teacher_topk": t_rank.cpu().tolist(),
        "student_argmax_ids": s_arg.numpy().tolist(),
        "teacher_argmax_ids": t_arg.numpy().tolist(),
        "top1_agree": top1_agree.numpy().tolist(),
        "topk_hit": topk_hit,
        "topk_kl": {name: vals.numpy().tolist() for name, vals in topk.items()},
    }


@torch.no_grad()
def score_rollout_batch(
    student_model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,
    tokenizer: Any,
    *,
    student_prompts: list[str],
    teacher_prompts: list[str],
    completion_ids_list: list[list[int]],
    temperature: float,
    topk_ks: tuple[int, ...] = (1, 16),
    topk_hit_ks: tuple[int, ...] = DEFAULT_TOPK_HIT_KS,
    jsd_beta: float = DEFAULT_JSD_BETA,
    jsd_token_clip: float = DEFAULT_JSD_TOKEN_CLIP,
) -> list[dict[str, Any]]:
    """Batched forward (student + teacher), per-rollout metrics on GPU one at a time."""
    if not student_prompts:
        return []
    device = student_model.device
    s_logits_l = forward_completion_logits(student_model, tokenizer, student_prompts, completion_ids_list)
    t_logits_l = forward_completion_logits(teacher_model, tokenizer, teacher_prompts, completion_ids_list)

    results: list[dict[str, Any]] = []
    for s_cpu, t_cpu, cids in zip(s_logits_l, t_logits_l, completion_ids_list):
        s_logits = s_cpu.to(device)
        t_logits = t_cpu.to(device)
        results.append(
            _metrics_from_logits(
                s_logits,
                t_logits,
                cids,
                temperature=temperature,
                topk_ks=topk_ks,
                topk_hit_ks=topk_hit_ks,
                jsd_beta=jsd_beta,
                jsd_token_clip=jsd_token_clip,
            )
        )
        del s_logits, t_logits
    return results


@torch.no_grad()
def score_rollout_pair(
    student_model: AutoModelForCausalLM,
    teacher_model: AutoModelForCausalLM,
    tokenizer: Any,
    *,
    student_prompt: str,
    teacher_prompt: str,
    completion_ids: list[int],
    temperature: float,
    topk_ks: tuple[int, ...] = (1, 16),
    topk_hit_ks: tuple[int, ...] = DEFAULT_TOPK_HIT_KS,
    jsd_beta: float = DEFAULT_JSD_BETA,
    jsd_token_clip: float = DEFAULT_JSD_TOKEN_CLIP,
) -> dict[str, Any]:
    """Compute per-position metrics for one rollout."""
    return score_rollout_batch(
        student_model,
        teacher_model,
        tokenizer,
        student_prompts=[student_prompt],
        teacher_prompts=[teacher_prompt],
        completion_ids_list=[completion_ids],
        temperature=temperature,
        topk_ks=topk_ks,
        topk_hit_ks=topk_hit_ks,
        jsd_beta=jsd_beta,
        jsd_token_clip=jsd_token_clip,
    )[0]


def apply_position_mask(metrics: dict[str, Any], mask: torch.Tensor) -> dict[str, Any]:
    """Filter per-token lists by boolean mask."""
    if metrics.get("n_tokens", 0) == 0:
        return metrics
    m = mask.cpu().numpy().astype(bool)
    out = dict(metrics)
    for key in PER_TOKEN_SCALAR_KEYS:
        if key not in metrics:
            continue
        out[key] = [v for v, keep in zip(metrics[key], m) if keep]
    for key in PER_TOKEN_ID_KEYS:
        if key not in metrics:
            continue
        out[key] = [v for v, keep in zip(metrics[key], m) if keep]
    out["topk_kl"] = {
        k: [v for v, keep in zip(vals, m) if keep]
        for k, vals in metrics.get("topk_kl", {}).items()
    }
    if "topk_hit" in metrics:
        out["topk_hit"] = {
            k: {
                side: [v for v, keep in zip(vals, m) if keep]
                for side, vals in sides.items()
            }
            for k, sides in metrics["topk_hit"].items()
        }
    out["n_tokens"] = len(out.get("token_ids", []))
    return out


def apply_length_window(metrics: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    n = metrics.get("n_tokens", 0)
    mask = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        if start <= i < end:
            mask[i] = True
    return apply_position_mask(metrics, mask)


def apply_entropy_bucket(metrics: dict[str, Any], bucket: str) -> dict[str, Any]:
    from .model_registry import entropy_ratio

    if metrics.get("n_tokens", 0) == 0:
        return metrics
    kind, ratio = entropy_ratio(bucket)
    ent = torch.tensor(metrics["student_entropy"], dtype=torch.float32)
    mask = entropy_mask(ent, ratio, high=(kind == "high"))
    return apply_position_mask(metrics, mask)


def _span_mean(values: list[float], start: int, end: int) -> float | None:
    if not values or start >= len(values):
        return None
    chunk = values[start:end]
    return float(sum(chunk) / len(chunk)) if chunk else None


def rollout_metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Per-rollout scalar summary for rollout_metrics.jsonl."""
    n = metrics.get("n_tokens", 0)
    if n == 0:
        return {"n_tokens": 0}

    jsd = metrics["jsd_kl"]
    jsd_clip = metrics.get("jsd_kl_clipped", jsd)
    adv = metrics.get("advantage") or [-x for x in metrics["log_ratio_k1"]]
    n_enc = sum(1 for a in adv if a > 0)
    n_dec = sum(1 for a in adv if a < 0)
    n_clip = sum(1 for x in metrics.get("jsd_would_clip", []) if x > 0)
    topk_hit = metrics.get("topk_hit", {})
    k_split = str(max(int(k) for k in topk_hit)) if topk_hit else "64"
    t_hit = topk_hit.get(k_split, {}).get("teacher", [])

    out: dict[str, Any] = {
        "n_tokens": n,
        "mean_jsd_kl": sum(jsd) / n,
        "max_jsd_kl": max(jsd),
        "mean_jsd_kl_clipped": sum(jsd_clip) / n,
        "frac_jsd_clipped": n_clip / n,
        "mean_advantage": sum(adv) / n,
        "max_advantage": max(adv),
        "min_advantage": min(adv),
        "frac_encourage": n_enc / n,
        "frac_discourage": n_dec / n,
        "mean_student_entropy": sum(metrics.get("student_entropy", [])) / n,
        "mean_teacher_entropy": sum(metrics.get("teacher_entropy", [])) / n,
        "mean_entropy_gap": sum(metrics.get("entropy_gap", [])) / n,
        "mean_student_confidence_gap": sum(metrics.get("student_confidence_gap", [])) / n,
        "mean_teacher_confidence_gap": sum(metrics.get("teacher_confidence_gap", [])) / n,
        "top1_agree_rate": sum(metrics.get("top1_agree", [])) / n,
        "first_128_mean_advantage": _span_mean(adv, 0, 128),
        "last_128_mean_advantage": _span_mean(adv, max(0, n - 128), n),
    }
    if t_hit:
        hit_n = sum(1 for h in t_hit if h)
        out[f"teacher_top{k_split}_hit_rate"] = hit_n / n
        hit_adv = [a for a, h in zip(adv, t_hit) if h]
        miss_adv = [a for a, h in zip(adv, t_hit) if not h]
        out["mean_advantage_teacher_hit"] = sum(hit_adv) / len(hit_adv) if hit_adv else None
        out["mean_advantage_teacher_miss"] = sum(miss_adv) / len(miss_adv) if miss_adv else None
    return out


def token_metrics_record(metrics: dict[str, Any], *, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten one rollout's per-token fields for jsonl persistence."""
    rec = {
        "n_tokens": metrics.get("n_tokens", 0),
        "token_ids": metrics.get("token_ids", []),
        "jsd_kl": metrics.get("jsd_kl", []),
        "jsd_kl_clipped": metrics.get("jsd_kl_clipped", []),
        "jsd_would_clip": metrics.get("jsd_would_clip", []),
        "log_ratio_k1": metrics.get("log_ratio_k1", []),
        "advantage": metrics.get("advantage", []),
        "student_entropy": metrics.get("student_entropy", []),
        "teacher_entropy": metrics.get("teacher_entropy", []),
        "entropy_gap": metrics.get("entropy_gap", []),
        "snr": metrics.get("snr", []),
        "student_logp": metrics.get("student_logp", []),
        "teacher_logp": metrics.get("teacher_logp", []),
        "student_confidence_gap": metrics.get("student_confidence_gap", []),
        "teacher_confidence_gap": metrics.get("teacher_confidence_gap", []),
        "rank_student_topk": metrics.get("rank_student_topk", []),
        "rank_teacher_topk": metrics.get("rank_teacher_topk", []),
        "student_argmax_ids": metrics.get("student_argmax_ids", []),
        "teacher_argmax_ids": metrics.get("teacher_argmax_ids", []),
        "top1_agree": metrics.get("top1_agree", []),
        "topk_hit": metrics.get("topk_hit", {}),
        "topk_kl": metrics.get("topk_kl", {}),
    }
    if meta:
        rec = {**meta, **rec}
    return rec
