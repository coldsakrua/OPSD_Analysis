"""Teacher-student divergence metrics aligned with OPSD training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

# train_opsd.py default --beta=0.0 (forward KL). The "0.05" in jsd005 is jsd_token_clip.
DEFAULT_JSD_BETA = 0.0
DEFAULT_JSD_TOKEN_CLIP = 0.05
DEFAULT_TOPK_HIT_KS = (4, 8, 16, 32, 64)
SNR_EPS = 1e-8


def compute_token_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Per-position Shannon entropy. logits: [L, V]."""
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def compute_jsd_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    beta: float = DEFAULT_JSD_BETA,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generalized JSD / KL per position (matches opsd_trainer.generalized_jsd_loss).

    Training default beta=0.0 → forward KL (teacher ‖ student) via F.kl_div swap.
    beta=1.0 → reverse KL (student ‖ teacher).
    beta=0.5 is symmetric JSD.

    Args:
        student_logits, teacher_logits: [L, V]
    Returns:
        [L] per-token divergence (sum over vocab dim already done)
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


def argmax_token_ids(logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    """Per-position preferred token = argmax of temperature-scaled logits."""
    return (logits / temperature).argmax(dim=-1)


def compute_snr(
    advantage: torch.Tensor,
    student_entropy: torch.Tensor,
    *,
    eps: float = SNR_EPS,
) -> torch.Tensor:
    """Per-token SNR = |log π_T(x) − log π_S(x)| / (H(π_S) + eps).

    Advantage follows the preference script: log π_T(x) − log π_S(x).
    High SNR ⇒ strong teacher/student disagreement relative to student uncertainty.
    """
    return advantage.abs() / (student_entropy + eps)


def sampled_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """log π(x) for sampled tokens at each position. logits [L,V], token_ids [L]."""
    scaled = logits / temperature
    logp = F.log_softmax(scaled, dim=-1)
    return logp.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def confidence_gap(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """max log π − log π(x): how far sampled token is from the model's top choice."""
    scaled = logits / temperature
    logp = F.log_softmax(scaled, dim=-1)
    tok_logp = logp.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return logp.max(dim=-1).values - tok_logp


def compute_topk_hit_and_ranks(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
    ks: tuple[int, ...] = DEFAULT_TOPK_HIT_KS,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Top-k membership for sampled token and 1-indexed rank (0 if outside max_k)."""
    scaled = logits / temperature
    max_k = max(ks)
    vocab = scaled.shape[-1]
    k_use = min(max_k, vocab)
    topk_idx = torch.topk(scaled, k=k_use, dim=-1).indices
    tok = token_ids.unsqueeze(-1)
    eq = topk_idx == tok
    in_max = eq.any(dim=-1)
    first = eq.float().argmax(dim=-1)
    rank = torch.where(in_max, first + 1, torch.zeros_like(first, dtype=torch.long))

    hits: dict[int, torch.Tensor] = {}
    for k in ks:
        kk = min(k, k_use)
        hits[k] = (topk_idx[:, :kk] == tok).any(dim=-1)
    return hits, rank


def clip_jsd_per_token(jsd: torch.Tensor, token_clip: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (clipped jsd, would_clip indicator as float 0/1)."""
    clipped = jsd.clamp(max=token_clip)
    would_clip = (jsd > token_clip).to(jsd.dtype)
    return clipped, would_clip


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
