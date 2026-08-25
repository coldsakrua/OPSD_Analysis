"""Forward-pass scoring and metric extraction."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from .metrics import (
    compute_jsd_kl,
    compute_token_entropy,
    compute_topk_kl,
    entropy_mask,
    log_ratio_k1,
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
) -> dict[str, Any]:
    """Compute per-position metrics for one rollout."""
    s_logits_l = forward_completion_logits(student_model, tokenizer, [student_prompt], [completion_ids])[0]
    t_logits_l = forward_completion_logits(teacher_model, tokenizer, [teacher_prompt], [completion_ids])[0]
    L = min(s_logits_l.shape[0], t_logits_l.shape[0], len(completion_ids))
    if L == 0:
        return {"n_tokens": 0}

    s_logits = s_logits_l[:L].to(student_model.device)
    t_logits = t_logits_l[:L].to(teacher_model.device)
    tok = torch.tensor(completion_ids[:L], dtype=torch.long, device=student_model.device)

    jsd = compute_jsd_kl(s_logits, t_logits, temperature=temperature).cpu()
    lr1 = log_ratio_k1(s_logits, t_logits, tok, temperature=temperature).cpu()
    ent = compute_token_entropy(s_logits, temperature=temperature).cpu()
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
        "log_ratio_k1": lr1.numpy().tolist(),
        "student_entropy": ent.numpy().tolist(),
        "topk_kl": {name: vals.numpy().tolist() for name, vals in topk.items()},
    }


def apply_position_mask(metrics: dict[str, Any], mask: torch.Tensor) -> dict[str, Any]:
    """Filter per-token lists by boolean mask."""
    if metrics.get("n_tokens", 0) == 0:
        return metrics
    m = mask.cpu().numpy().astype(bool)
    out = dict(metrics)
    for key in ("jsd_kl", "log_ratio_k1", "student_entropy"):
        arr = metrics[key]
        out[key] = [v for v, keep in zip(arr, m) if keep]
    out["topk_kl"] = {
        k: [v for v, keep in zip(vals, m) if keep]
        for k, vals in metrics["topk_kl"].items()
    }
    out["token_ids"] = [v for v, keep in zip(metrics["token_ids"], m) if keep]
    out["n_tokens"] = len(out["token_ids"])
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
