"""Teacher-student divergence metrics aligned with OPSD training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_token_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Per-position Shannon entropy. logits: [L, V]."""
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def compute_jsd_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generalized JSD KL per position (matches opsd_trainer.generalized_jsd_loss, beta=0.5).

    Args:
        student_logits, teacher_logits: [L, V]
    Returns:
        [L] per-token JSD values (sum over vocab dim already done)
    """
    s_logits = student_logits / temperature
    t_logits = teacher_logits / temperature
    s_logp = F.log_softmax(s_logits, dim=-1)
    t_logp = F.log_softmax(t_logits, dim=-1)

    if beta == 0:
        return F.kl_div(s_logp, t_logp, reduction="none", log_target=True).sum(dim=-1)
    if beta == 1:
        return F.kl_div(t_logp, s_logp, reduction="none", log_target=True).sum(dim=-1)

    b = torch.tensor(beta, dtype=s_logp.dtype, device=s_logp.device)
    mixture = torch.logsumexp(
        torch.stack([s_logp + torch.log1p(-b), t_logp + torch.log(b)]),
        dim=0,
    )
    kl_t = F.kl_div(mixture, t_logp, reduction="none", log_target=True).sum(dim=-1)
    kl_s = F.kl_div(mixture, s_logp, reduction="none", log_target=True).sum(dim=-1)
    return b * kl_t + (1 - b) * kl_s


def compute_topk_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    k: int,
    temperature: float = 1.0,
    direction: str = "teacher_to_student",
) -> torch.Tensor:
    """OPD-style top-k KL: restrict to teacher top-k, renormalize, D_KL(pt || qs).

    Per-position during forward pass.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    s_logits = student_logits / temperature
    t_logits = teacher_logits / temperature
    _, top_idx = torch.topk(t_logits, k=min(k, t_logits.shape[-1]), dim=-1)
    s_sub = torch.gather(s_logits, dim=-1, index=top_idx)
    t_sub = torch.gather(t_logits, dim=-1, index=top_idx)
    s_logp = F.log_softmax(s_sub, dim=-1)
    t_logp = F.log_softmax(t_sub, dim=-1)
    if direction == "teacher_to_student":
        return F.kl_div(s_logp, t_logp, reduction="none", log_target=True).sum(dim=-1)
    return F.kl_div(t_logp, s_logp, reduction="none", log_target=True).sum(dim=-1)


def log_ratio_k1(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """log π_S(x) - log π_T(x) for sampled token x at each position."""
    s_logits = student_logits / temperature
    t_logits = teacher_logits / temperature
    s_logp = F.log_softmax(s_logits, dim=-1)
    t_logp = F.log_softmax(t_logits, dim=-1)
    s_tok = s_logp.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    t_tok = t_logp.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return s_tok - t_tok


def entropy_mask(
    entropy: torch.Tensor,
    ratio: float,
    *,
    high: bool,
) -> torch.Tensor:
    """Select top/bottom-ρ entropy positions within a 1-D entropy vector."""
    n = entropy.numel()
    if n == 0:
        return torch.zeros_like(entropy, dtype=torch.bool)
    if ratio >= 1.0:
        return torch.ones_like(entropy, dtype=torch.bool)
    k = max(1, int(torch.ceil(torch.tensor(ratio * n)).item()))
    _, idx = torch.topk(entropy, k=k, largest=high)
    mask = torch.zeros(n, dtype=torch.bool, device=entropy.device)
    mask[idx] = True
    return mask
