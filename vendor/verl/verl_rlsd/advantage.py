from __future__ import annotations

import logging
import math
from typing import Any, Iterable

import torch

from .cast_verify import log_cast_verify

logger = logging.getLogger(__name__)

# Monotonic counter so we can log sparsely after the first few calls.
_CAST_VERIFY_STEP: list[int] = [0]
_CAST_ADV_ENTRY_LOGGED: list[bool] = [False]

try:
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn, register_adv_est
except Exception:  # pragma: no cover - lets local syntax checks run without veRL.
    _LOCAL_ADV_REGISTRY = {}

    def register_adv_est(name: str):
        def deco(fn):
            _LOCAL_ADV_REGISTRY[name] = fn
            return fn

        return deco

    def get_adv_estimator_fn(name: str):
        return _LOCAL_ADV_REGISTRY[name]


CUSTOM_ADV_ESTIMATORS = {
    "rlsd_grpo",
    "rlsd_strict_split_flip",
    "rlsd_strict_split_flip_wrong_boost",
    "rlsd_strict_split_preserve_sign",
    "rlrt",
    "opd_zero",
}


def _cfg_get(config: Any, path: str, default: Any = None) -> Any:
    cur = config
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


def get_custom_adv_estimator_name(config: Any) -> str | None:
    name = _cfg_get(config, "rlsd.custom_adv_estimator")
    if name is None:
        name = _cfg_get(config, "algorithm.rlsd.custom_adv_estimator")
    if name is None:
        return None
    text = str(name).strip()
    return text or None


def is_registered_adv_estimator(name: str) -> bool:
    return str(name) in CUSTOM_ADV_ESTIMATORS


def _teacher_src_label(kwargs: dict[str, Any]) -> str:
    if _first_not_none(
        kwargs.get("teacher_log_probs"),
        kwargs.get("teacher_logprobs"),
        _batch_get(
            kwargs.get("batch"),
            "teacher_log_probs",
            "teacher_logprobs",
            "teacher_log_probs_all",
        ),
    ) is not None:
        return "teacher_log_probs"
    if _first_not_none(
        kwargs.get("ref_log_prob"),
        kwargs.get("ref_log_probs"),
        _batch_get(kwargs.get("batch"), "ref_log_prob", "ref_log_probs"),
    ) is not None:
        return "ref_log_prob"
    return "zeros(fallback)"


_LATEST_CAST_TRAIN_METRICS: dict[str, float] = {}


def _record_cast_ratio_metrics(
    *,
    g: torch.Tensor,
    base_adv: torch.Tensor,
    teacher_weight: torch.Tensor,
    mask: torch.Tensor,
    correct_mask: torch.Tensor,
    correct_low: float,
    correct_high: float,
    wrong_low: float,
    wrong_high: float,
    teacher_src: str,
    lam: float,
) -> dict[str, float]:
    """Per-step CAST ratio / clip stats for wandb (via compute_data_metrics)."""
    with torch.no_grad():
        m = mask.bool()
        if not bool(m.any()):
            metrics = {
                "cast/teacher_student_ratio_mean": 1.0,
                "cast/teacher_student_gap_mean": 0.0,
                "cast/teacher_student_gap_abs_mean": 0.0,
                "cast/teacher_weight_mean": 1.0,
                "cast/teacher_weight_clip_frac": 0.0,
                "cast/token_gap_lambda": float(lam),
                "cast/teacher_src_is_teacher": 1.0 if teacher_src == "teacher_log_probs" else 0.0,
            }
            _LATEST_CAST_TRAIN_METRICS.clear()
            _LATEST_CAST_TRAIN_METRICS.update(metrics)
            return metrics

        g_vals = g[m]
        sign = torch.sign(base_adv)[m]
        raw_weight = _safe_exp_gap(sign * g_vals)
        is_correct = correct_mask.expand_as(g)[m].bool()
        low = torch.where(
            is_correct,
            torch.full_like(raw_weight, float(correct_low)),
            torch.full_like(raw_weight, float(wrong_low)),
        )
        high = torch.where(
            is_correct,
            torch.full_like(raw_weight, float(correct_high)),
            torch.full_like(raw_weight, float(wrong_high)),
        )
        clipped = (raw_weight < low) | (raw_weight > high)
        metrics = {
            "cast/teacher_student_ratio_mean": float(_safe_exp_gap(g_vals).mean().item()),
            "cast/teacher_student_gap_mean": float(g_vals.mean().item()),
            "cast/teacher_student_gap_abs_mean": float(g_vals.abs().mean().item()),
            "cast/teacher_weight_mean": float(teacher_weight[m].mean().item()),
            "cast/teacher_weight_clip_frac": float(clipped.float().mean().item()),
            "cast/token_gap_lambda": float(lam),
            "cast/teacher_src_is_teacher": 1.0 if teacher_src == "teacher_log_probs" else 0.0,
        }
    _LATEST_CAST_TRAIN_METRICS.clear()
    _LATEST_CAST_TRAIN_METRICS.update(metrics)
    return metrics


def _teacher_weight_from_gap(
    base_adv: torch.Tensor,
    g: torch.Tensor,
    mask: torch.Tensor,
    correct_mask: torch.Tensor,
    *,
    correct_low: float,
    correct_high: float,
    wrong_low: float,
    wrong_high: float,
) -> torch.Tensor:
    """CAST token weight.

    g = log(pT/pS)
    weight = clip(exp(sign(A) * g), traj-asymmetric bounds)

    - Correct traj: raise-prob side clipped to +0.05 (high=1.05), suppress side to -0.2 (low=0.8)
    - Wrong traj: raise-prob side clipped to -0.05 (low=0.95), suppress side to +0.2 (high=1.2)
    """
    sign = torch.sign(base_adv)
    weight = _safe_exp_gap(sign * g)
    is_correct = correct_mask.expand_as(g).bool()
    low = torch.where(
        is_correct,
        torch.full_like(weight, float(correct_low)),
        torch.full_like(weight, float(wrong_low)),
    )
    high = torch.where(
        is_correct,
        torch.full_like(weight, float(correct_high)),
        torch.full_like(weight, float(wrong_high)),
    )
    return torch.minimum(torch.maximum(weight, low), high) * mask


def _reward_group_stats(
    token_level_rewards: torch.Tensor,
    mask: torch.Tensor,
    config: Any,
) -> dict[str, Any] | None:
    rewards_binary = _binary_correct(
        token_level_rewards,
        float(_cfg_get(config, "rlsd.reward_binary_threshold", 0.5)),
    )
    num_generations = int(_cfg_get(config, "actor_rollout_ref.rollout.n", 8) or 8)
    if rewards_binary.numel() % num_generations != 0:
        return None
    grouped = rewards_binary.view(-1, num_generations)
    all_correct_group = (grouped > 0.5).all(dim=1)
    all_wrong_group = (grouped < 0.5).all(dim=1)
    mixed_group = ~(all_correct_group | all_wrong_group)
    all_correct = all_correct_group.repeat_interleave(num_generations).unsqueeze(1)
    all_wrong = all_wrong_group.repeat_interleave(num_generations).unsqueeze(1)
    mixed = mixed_group.repeat_interleave(num_generations).unsqueeze(1)
    return {
        "grouped": grouped,
        "all_correct_group": all_correct_group,
        "all_wrong_group": all_wrong_group,
        "mixed_group": mixed_group,
        "all_correct": all_correct,
        "all_wrong": all_wrong,
        "mixed": mixed,
    }


def _maybe_log_teacher_weight_verify(
    *,
    estimator: str,
    token_level_rewards: torch.Tensor,
    kwargs: dict[str, Any],
    g: torch.Tensor,
    grpo_base: torch.Tensor,
    shaped_pre_clip: torch.Tensor,
    final_adv: torch.Tensor,
    teacher_weight: torch.Tensor,
    mask: torch.Tensor,
    lam: float,
    extra: dict[str, Any] | None = None,
) -> None:
    _CAST_VERIFY_STEP[0] += 1
    step = _CAST_VERIFY_STEP[0]
    if step > 5 and step % 50 != 0:
        return
    stats = _reward_group_stats(token_level_rewards, mask, kwargs.get("config"))
    if stats is None:
        n_rows = int(mask.shape[0])
        all_correct = torch.zeros(n_rows, 1, dtype=torch.bool)
        all_wrong = torch.zeros(n_rows, 1, dtype=torch.bool)
        mixed = torch.ones(n_rows, 1, dtype=torch.bool)
        grouped_counts = {
            "all_correct": 0.0,
            "all_wrong": 0.0,
            "mixed": float(n_rows),
        }
    else:
        all_correct = stats["all_correct"]
        all_wrong = stats["all_wrong"]
        mixed = stats["mixed"]
        n_total = max(float(stats["grouped"].shape[0]), 1.0)
        grouped_counts = {
            "all_correct": float(stats["all_correct_group"].sum().item()),
            "all_wrong": float(stats["all_wrong_group"].sum().item()),
            "mixed": float(stats["mixed_group"].sum().item()),
            "all_correct_pct": 100.0 * float(stats["all_correct_group"].sum().item()) / n_total,
            "all_wrong_pct": 100.0 * float(stats["all_wrong_group"].sum().item()) / n_total,
            "mixed_pct": 100.0 * float(stats["mixed_group"].sum().item()) / n_total,
        }
    log_cast_verify(
        step=step,
        estimator=estimator,
        grouped_counts=grouped_counts,
        teacher_src=_teacher_src_label(kwargs),
        g=g,
        grpo_base=grpo_base,
        shaped_pre_clip=shaped_pre_clip,
        final_adv=final_adv,
        weight=teacher_weight,
        mask=mask,
        all_correct=all_correct,
        all_wrong=all_wrong,
        mixed=mixed,
        lam=lam,
        fallback_scale=1.0,
        extra=extra,
    )


def _maybe_log_cast_verify(
    *,
    estimator: str,
    grouped: torch.Tensor,
    all_correct_group: torch.Tensor,
    all_wrong_group: torch.Tensor,
    mixed_group: torch.Tensor,
    all_correct: torch.Tensor,
    all_wrong: torch.Tensor,
    mixed: torch.Tensor,
    kwargs: dict[str, Any],
    g: torch.Tensor,
    grpo_base: torch.Tensor,
    shaped_pre_clip: torch.Tensor,
    final_adv: torch.Tensor,
    teacher_weight: torch.Tensor,
    mask: torch.Tensor,
    lam: float,
    fallback_scale: float,
    extra: dict[str, Any] | None = None,
) -> None:
    _CAST_VERIFY_STEP[0] += 1
    step = _CAST_VERIFY_STEP[0]
    if step > 5 and step % 50 != 0:
        return
    n_total = max(float(grouped.shape[0]), 1.0)
    log_cast_verify(
        step=step,
        estimator=estimator,
        grouped_counts={
            "all_correct": float(all_correct_group.sum().item()),
            "all_wrong": float(all_wrong_group.sum().item()),
            "mixed": float(mixed_group.sum().item()),
            "all_correct_pct": 100.0 * float(all_correct_group.sum().item()) / n_total,
            "all_wrong_pct": 100.0 * float(all_wrong_group.sum().item()) / n_total,
            "mixed_pct": 100.0 * float(mixed_group.sum().item()) / n_total,
        },
        teacher_src=_teacher_src_label(kwargs),
        g=g,
        grpo_base=grpo_base,
        shaped_pre_clip=shaped_pre_clip,
        final_adv=final_adv,
        weight=teacher_weight,
        mask=mask,
        all_correct=all_correct,
        all_wrong=all_wrong,
        mixed=mixed,
        lam=lam,
        fallback_scale=fallback_scale,
        extra=extra,
    )


def _as_tensor(value: Any, like: torch.Tensor | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        out = value
    else:
        try:
            out = torch.as_tensor(value)
        except Exception:
            return None
    if like is not None:
        out = out.to(device=like.device)
    return out


def _batch_get(batch: Any, *keys: str) -> Any:
    if batch is None:
        return None
    for key in keys:
        try:
            value = batch.get(key)
            if value is not None:
                return value
        except Exception:
            pass
        try:
            value = batch[key]
            if value is not None:
                return value
        except Exception:
            pass
    return None


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _response_width(response_mask: torch.Tensor) -> int:
    return int(response_mask.shape[-1])


def _last_response_logps(value: Any, response_mask: torch.Tensor) -> torch.Tensor | None:
    tensor = _last_response_slice(value, response_mask)
    if tensor is None:
        return None
    if tensor.dim() >= 3:
        tensor = tensor[..., 0]
    return tensor.to(dtype=torch.float32, device=response_mask.device)


def _last_response_slice(value: Any, response_mask: torch.Tensor) -> torch.Tensor | None:
    tensor = _as_tensor(value, response_mask)
    if tensor is None:
        return None
    width = _response_width(response_mask)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    response_dim = -2 if tensor.dim() >= 3 else -1
    if tensor.shape[response_dim] != width:
        if response_dim == -2:
            tensor = tensor[:, -width:, ...]
        else:
            tensor = tensor[..., -width:]
    return tensor.to(device=response_mask.device)


def _indices(index: Any, n: int) -> list[Any]:
    if index is None:
        return list(range(n))
    try:
        values = list(index)
    except TypeError:
        return list(range(n))
    out: list[Any] = []
    for value in values[:n]:
        if isinstance(value, torch.Tensor):
            value = value.item() if value.numel() == 1 else tuple(value.tolist())
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        out.append(value)
    if len(out) < n:
        out.extend(range(len(out), n))
    return out


def _grouped_grpo_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any = None,
    *,
    norm_by_std: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    scores = token_level_rewards.sum(dim=-1).float()
    n = int(scores.shape[0])
    ids = _indices(index, n)
    base = torch.zeros_like(scores)
    groups: dict[Any, list[int]] = {}
    for i, key in enumerate(ids):
        groups.setdefault(key, []).append(i)
    for members in groups.values():
        values = scores[members]
        mean = values.mean()
        if norm_by_std and len(members) > 1:
            std = values.std(unbiased=False).clamp_min(eps)
            base[members] = (values - mean) / std
        else:
            base[members] = values - mean
    return base.unsqueeze(-1).expand_as(response_mask).to(dtype=torch.float32) * response_mask


def _binary_correct(
    token_level_rewards: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    scores = token_level_rewards.sum(dim=-1).float()
    return (scores > float(threshold)).float()


def _lambda_decay_enabled(config: Any, prefix: str = "rlsd") -> bool:
    """Linear lambda decay is only for canonical RLSD (rlsd_grpo); CAST/RLRT stay constant."""
    flag = _cfg_get(config, f"{prefix}.token_gap_decay_enabled", None)
    if flag is not None:
        return bool(flag)
    estimator = _cfg_get(config, f"{prefix}.custom_adv_estimator")
    if estimator is None:
        estimator = _cfg_get(config, "algorithm.rlsd.custom_adv_estimator")
    if str(estimator or "").strip() != "rlsd_grpo":
        return False
    decay_steps = int(
        _cfg_get(
            config,
            f"{prefix}.token_gap_decay_steps",
            _cfg_get(config, f"{prefix}.lmbda_decay_steps", 0),
        )
        or 0
    )
    return decay_steps > 0


def _current_lambda(config: Any, prefix: str = "rlsd", *, global_step: int | None = None) -> float:
    lmbda = float(_cfg_get(config, f"{prefix}.token_gap_lambda", _cfg_get(config, f"{prefix}.lmbda", 1.0)))
    if not _lambda_decay_enabled(config, prefix):
        return lmbda
    decay_steps = int(_cfg_get(config, f"{prefix}.token_gap_decay_steps", _cfg_get(config, f"{prefix}.lmbda_decay_steps", 0)) or 0)
    if decay_steps <= 0:
        return lmbda
    step = global_step
    if step is None:
        step = int(_cfg_get(config, "trainer.global_step", 0) or 0)
    if step <= 0:
        try:
            from .teacher_ema import _get_trainer_global_step

            step = _get_trainer_global_step(0)
        except Exception:
            step = 0
    progress = min(max(int(step), 0), decay_steps) / float(decay_steps)
    return lmbda * (1.0 - progress)


def _teacher_student_gap(response_mask: torch.Tensor, kwargs: dict[str, Any]) -> torch.Tensor:
    batch = kwargs.get("batch")
    student = _last_response_logps(
        _first_not_none(
            kwargs.get("old_log_probs"),
            kwargs.get("old_log_prob"),
            _batch_get(batch, "old_log_probs", "old_log_prob"),
        ),
        response_mask,
    )
    teacher_raw = _first_not_none(
        kwargs.get("teacher_log_probs"),
        kwargs.get("teacher_logprobs"),
        _batch_get(batch, "teacher_log_probs", "teacher_logprobs", "teacher_log_probs_all"),
    )
    teacher_tensor = _last_response_slice(teacher_raw, response_mask)
    teacher = None
    if teacher_tensor is not None and teacher_tensor.dim() >= 3:
        teacher_ids = _last_response_slice(
            _batch_get(batch, "teacher_ids", "teacher_topk_ids"),
            response_mask,
        )
        response_ids = _last_response_slice(
            _batch_get(batch, "responses", "response_ids", "completion_ids"),
            response_mask,
        )
        if teacher_ids is not None and response_ids is not None:
            if response_ids.dim() >= 3:
                response_ids = response_ids[..., 0]
            matches = teacher_ids.long() == response_ids.long().unsqueeze(-1)
            has_match = matches.any(dim=-1)
            gathered = (teacher_tensor.float() * matches.float()).sum(dim=-1)
            teacher = torch.where(has_match, gathered, teacher_tensor[..., 0].float())
        else:
            teacher = teacher_tensor[..., 0].float()
    elif teacher_tensor is not None:
        teacher = teacher_tensor.float()
    if teacher is None:
        teacher = _last_response_logps(
            _first_not_none(
                kwargs.get("ref_log_prob"),
                _batch_get(batch, "ref_log_prob", "ref_log_probs"),
            ),
            response_mask,
        )
    if student is None or teacher is None:
        return torch.zeros_like(response_mask, dtype=torch.float32)
    return (teacher - student).detach() * response_mask.float()


def _teacher_shaping_length_cap(config: Any) -> int:
    cap = int(_cfg_get(config, "rlsd.teacher_shaping_length_cap", 0) or 0)
    if cap > 0:
        return cap
    # Backward-compatible alias: older scripts only set max_teacher_prompt_length.
    return int(_cfg_get(config, "rlsd.max_teacher_prompt_length", 0) or 0)


def _blend_teacher_shaping(
    base_adv: torch.Tensor,
    shaped_adv: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_shaping_length_cap: int,
) -> torch.Tensor:
    """Keep GRPO/base A outside the cap; apply teacher RLSD shaping only inside."""
    if teacher_shaping_length_cap <= 0:
        return shaped_adv
    mask = response_mask.float()
    response_pos = torch.cumsum(mask, dim=-1)
    within_cap = (response_pos <= float(teacher_shaping_length_cap)) & mask.bool()
    return torch.where(within_cap, shaped_adv, base_adv)


def _safe_exp_gap(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(x, min=-20.0, max=20.0))


def _zero_crossed_advantages(
    shaped_adv: torch.Tensor,
    base_adv: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    crossed = ((base_adv > 0) & (shaped_adv < 0)) | ((base_adv < 0) & (shaped_adv > 0))
    return torch.where(crossed, torch.zeros_like(shaped_adv), shaped_adv) * mask.float()


@register_adv_est("opd_zero")
def opd_zero_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any = None,
    config: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    adv = torch.zeros_like(response_mask, dtype=torch.float32)
    return adv, adv


@register_adv_est("rlsd_grpo")
def rlsd_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any = None,
    config: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm = bool(_cfg_get(config, "norm_adv_by_std_in_grpo", True))
    base = _grouped_grpo_advantages(token_level_rewards, response_mask.float(), index, norm_by_std=norm)
    lam = _current_lambda(config, global_step=kwargs.get("global_step"))
    eps_w = float(_cfg_get(config, "rlsd.jsd_token_clip", 0.2))
    g = _teacher_student_gap(response_mask, kwargs)
    sign = torch.sign(base)
    weight = _safe_exp_gap(sign * g)
    weight = torch.clamp(weight, min=max(0.0, 1.0 - eps_w), max=1.0 + eps_w)
    shaped = base * ((1.0 - lam) + lam * weight)
    mask = response_mask.float()
    shaped = _blend_teacher_shaping(
        base,
        shaped,
        mask,
        _teacher_shaping_length_cap(config),
    )
    shaped = shaped * mask
    teacher_weight = torch.clamp(weight, min=max(0.0, 1.0 - eps_w), max=1.0 + eps_w) * mask
    _maybe_log_teacher_weight_verify(
        estimator="rlsd_grpo",
        token_level_rewards=token_level_rewards,
        kwargs=kwargs,
        g=g,
        grpo_base=base,
        shaped_pre_clip=shaped,
        final_adv=shaped,
        teacher_weight=teacher_weight,
        mask=mask,
        lam=lam,
        extra={
            "algorithm_core": "rlsd_grpo",
            "strict_split": False,
            "teacher_weight_on": "all_tokens",
            "grpo_fallback": False,
        },
    )
    return shaped, shaped


def _strict_split_common(
    *,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any,
    config: Any,
    wrong_boost: bool,
    kwargs: dict[str, Any],
    preserve_sign: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = response_mask.float()
    norm = bool(_cfg_get(config, "norm_adv_by_std_in_grpo", True))
    seq_adv = _grouped_grpo_advantages(token_level_rewards, mask, index, norm_by_std=norm)
    seq_adv_1d = (seq_adv * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    rewards_binary = _binary_correct(
        token_level_rewards,
        float(_cfg_get(config, "rlsd.reward_binary_threshold", 0.5)),
    )
    num_generations = int(_cfg_get(config, "actor_rollout_ref.rollout.n", 8) or 8)
    if rewards_binary.numel() % num_generations != 0:
        return seq_adv, seq_adv

    grouped = rewards_binary.view(-1, num_generations)
    all_correct_group = (grouped > 0.5).all(dim=1)
    all_wrong_group = (grouped < 0.5).all(dim=1)
    mixed_group = ~(all_correct_group | all_wrong_group)
    all_correct = all_correct_group.repeat_interleave(num_generations).unsqueeze(1)
    all_wrong = all_wrong_group.repeat_interleave(num_generations).unsqueeze(1)
    mixed = mixed_group.repeat_interleave(num_generations).unsqueeze(1)

    lam = _current_lambda(config, global_step=kwargs.get("global_step"))
    token_gap_lambda = float(_cfg_get(config, "rlsd.token_gap_lambda", _cfg_get(config, "rlsd.lmbda", 1.0)))
    if abs(token_gap_lambda) <= 1e-12:
        fallback_scale = 0.0
    else:
        fallback_scale = min(max(abs(lam / token_gap_lambda), 0.0), 1.0)

    g = _teacher_student_gap(response_mask, kwargs)
    ratio_deadband_low = float(_cfg_get(config, "rlsd.token_ratio_deadband_low", 1.0) or 1.0)
    ratio_deadband_high = float(_cfg_get(config, "rlsd.token_ratio_deadband_high", 1.0) or 1.0)
    if ratio_deadband_low > 0.0 and ratio_deadband_high > ratio_deadband_low:
        teacher_student_ratio = _safe_exp_gap(g)
        active_gap = (teacher_student_ratio < ratio_deadband_low) | (
            teacher_student_ratio > ratio_deadband_high
        )
        g = torch.where(active_gap, g, torch.zeros_like(g))
    # weight = clip(exp(sign(A)*g)); bounds near 1, asymmetric by traj correctness.
    # Correct: [0.8, 1.05] (suppress ±0.2, raise ±0.05). Wrong: [0.95, 1.2].
    correct_low = float(_cfg_get(config, "rlsd.correct_weight_clip_low", 0.8))
    correct_high = float(_cfg_get(config, "rlsd.correct_weight_clip_high", 1.05))
    wrong_low = float(_cfg_get(config, "rlsd.wrong_weight_clip_low", 0.95))
    wrong_high = float(_cfg_get(config, "rlsd.wrong_weight_clip_high", 1.2))
    adv_low = float(_cfg_get(config, "rlsd.adv_clip_low", -1.0e9))
    adv_high = float(_cfg_get(config, "rlsd.adv_clip_high", 1.0e9))
    teacher_shaping_length_cap = _teacher_shaping_length_cap(config)
    # Per-trajectory correctness: selects asymmetric clip box (not the weight direction).
    correct_mask = (rewards_binary > 0.5).unsqueeze(1)

    def shape(base_adv: torch.Tensor) -> torch.Tensor:
        # A' = A * (1 + lambda * (weight - 1)); lambda=1 => A' = A * weight.
        weight = _teacher_weight_from_gap(
            base_adv,
            g,
            mask,
            correct_mask,
            correct_low=correct_low,
            correct_high=correct_high,
            wrong_low=wrong_low,
            wrong_high=wrong_high,
        )
        return base_adv * torch.clamp(1.0 + lam * (weight - 1.0) * mask, min=0.0) * mask

    def teacher_weight_for(base_adv: torch.Tensor) -> torch.Tensor:
        return _teacher_weight_from_gap(
            base_adv,
            g,
            mask,
            correct_mask,
            correct_low=correct_low,
            correct_high=correct_high,
            wrong_low=wrong_low,
            wrong_high=wrong_high,
        )

    def apply_teacher_shaping(base_adv: torch.Tensor) -> torch.Tensor:
        return _blend_teacher_shaping(base_adv, shape(base_adv), mask, teacher_shaping_length_cap)

    mixed_base = seq_adv_1d.unsqueeze(1).expand_as(mask) * mask
    all_correct_base = (
        torch.full_like(mask, float(_cfg_get(config, "rlsd.all_correct_base_advantage", 1.0)))
        * fallback_scale
        * mask
    )
    all_wrong_base = (
        torch.full_like(mask, float(_cfg_get(config, "rlsd.all_wrong_base_advantage", -1.0)))
        * fallback_scale
        * mask
    )

    mixed_only = bool(_cfg_get(config, "rlsd.mixed_only", False))

    base_token_adv = torch.zeros_like(mask)
    if mixed_only:
        base_token_adv = torch.where(mixed, mixed_base, base_token_adv)
    else:
        base_token_adv = torch.where(all_correct, all_correct_base, base_token_adv)
        base_token_adv = torch.where(all_wrong, all_wrong_base, base_token_adv)
        base_token_adv = torch.where(mixed, mixed_base, base_token_adv)

    token_adv = torch.zeros_like(mask)
    teacher_weight = torch.zeros_like(mask)
    if mixed_only:
        token_adv = torch.where(mixed, apply_teacher_shaping(mixed_base), token_adv)
        teacher_weight = torch.where(mixed, teacher_weight_for(mixed_base), teacher_weight)
    else:
        token_adv = torch.where(all_correct, apply_teacher_shaping(all_correct_base), token_adv)
        token_adv = torch.where(all_wrong, apply_teacher_shaping(all_wrong_base), token_adv)
        token_adv = torch.where(mixed, apply_teacher_shaping(mixed_base), token_adv)
        teacher_weight = torch.where(all_correct, teacher_weight_for(all_correct_base), teacher_weight)
        teacher_weight = torch.where(all_wrong, teacher_weight_for(all_wrong_base), teacher_weight)
        teacher_weight = torch.where(mixed, teacher_weight_for(mixed_base), teacher_weight)

    shaped_pre_clip = token_adv.clone()

    positive_adv_length_cap = int(_cfg_get(config, "rlsd.positive_adv_length_cap", 0) or 0)
    if positive_adv_length_cap > 0:
        response_pos = torch.cumsum(mask, dim=-1)
        past_cap = (response_pos > float(positive_adv_length_cap)) & mask.bool()
        token_adv = torch.where(
            past_cap & (token_adv > 0),
            torch.zeros_like(token_adv),
            token_adv,
        )

    length_penalty_start = int(_cfg_get(config, "rlsd.length_penalty_start", 0) or 0)
    length_penalty = max(0.0, float(_cfg_get(config, "rlsd.length_penalty", 0.0) or 0.0))
    if length_penalty_start > 0 and length_penalty > 0.0:
        response_pos = torch.cumsum(mask, dim=-1)
        past_penalty_start = (response_pos > float(length_penalty_start)) & mask.bool()
        token_adv = token_adv - length_penalty * past_penalty_start.float()

    token_adv = torch.clamp(token_adv, min=adv_low, max=adv_high) * mask
    if preserve_sign:
        token_adv = _zero_crossed_advantages(token_adv, base_token_adv, mask)

    estimator = "rlsd_strict_split_preserve_sign" if preserve_sign else (
        "rlsd_strict_split_flip_wrong_boost" if wrong_boost else "rlsd_strict_split_flip"
    )
    _record_cast_ratio_metrics(
        g=g,
        base_adv=base_token_adv,
        teacher_weight=teacher_weight,
        mask=mask,
        correct_mask=correct_mask,
        correct_low=correct_low,
        correct_high=correct_high,
        wrong_low=wrong_low,
        wrong_high=wrong_high,
        teacher_src=_teacher_src_label(kwargs),
        lam=lam,
    )
    _maybe_log_cast_verify(
        estimator=estimator,
        grouped=grouped,
        all_correct_group=all_correct_group,
        all_wrong_group=all_wrong_group,
        mixed_group=mixed_group,
        all_correct=all_correct,
        all_wrong=all_wrong,
        mixed=mixed,
        kwargs=kwargs,
        g=g,
        grpo_base=base_token_adv,
        shaped_pre_clip=shaped_pre_clip,
        final_adv=token_adv,
        teacher_weight=teacher_weight,
        mask=mask,
        lam=lam,
        fallback_scale=fallback_scale,
        extra={
            "algorithm_core": estimator,
            "strict_split": True,
            "grpo_fallback": False,
        },
    )
    return token_adv, token_adv
def rlsd_strict_split_flip_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any = None,
    config: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _strict_split_common(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config=config,
        wrong_boost=False,
        kwargs=kwargs,
    )


@register_adv_est("rlsd_strict_split_flip_wrong_boost")
def rlsd_strict_split_flip_wrong_boost_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any = None,
    config: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _strict_split_common(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config=config,
        wrong_boost=True,
        kwargs=kwargs,
    )


@register_adv_est("rlsd_strict_split_preserve_sign")
def rlsd_strict_split_preserve_sign_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any = None,
    config: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _strict_split_common(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config=config,
        wrong_boost=False,
        preserve_sign=True,
        kwargs=kwargs,
    )


@register_adv_est("rlrt")
def rlrt_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Any = None,
    config: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = response_mask.float()
    norm = bool(_cfg_get(config, "norm_adv_by_std_in_grpo", True))
    base = _grouped_grpo_advantages(token_level_rewards, mask, index, norm_by_std=norm)
    rewards_binary = _binary_correct(
        token_level_rewards,
        float(_cfg_get(config, "rlsd.reward_binary_threshold", 0.5)),
    ).unsqueeze(1)
    lam = _current_lambda(config, global_step=kwargs.get("global_step"))
    eps_w = max(0.0, float(_cfg_get(config, "rlsd.rlrt_weight_clip", 1.0)))
    # RLRT uses student minus teacher, the reverse of RLSD.
    d_hat = -_teacher_student_gap(response_mask, kwargs)
    weight = _safe_exp_gap(torch.sign(base) * d_hat)
    weight = torch.clamp(weight, min=max(0.0, 1.0 - eps_w), max=1.0 + eps_w)
    shaped = base * ((1.0 - lam) + lam * weight)
    shaped = shaped * mask
    teacher_weight = weight * mask
    correct = rewards_binary > 0.5
    token_adv = torch.where(correct, shaped, base) * mask
    adv_low = float(_cfg_get(config, "rlsd.adv_clip_low", -1.0e9))
    adv_high = float(_cfg_get(config, "rlsd.adv_clip_high", 1.0e9))
    token_adv = torch.clamp(token_adv, min=adv_low, max=adv_high) * mask
    correct_traj_pct = 100.0 * float(correct.float().mean().item())
    _maybe_log_teacher_weight_verify(
        estimator="rlrt",
        token_level_rewards=token_level_rewards,
        kwargs=kwargs,
        g=d_hat,
        grpo_base=base,
        shaped_pre_clip=shaped,
        final_adv=token_adv,
        teacher_weight=teacher_weight,
        mask=mask,
        lam=lam,
        extra={
            "algorithm_core": "rlrt",
            "strict_split": False,
            "teacher_weight_on": "correct_trajectories_only",
            "correct_traj_pct": correct_traj_pct,
            "grpo_fallback": False,
        },
    )
    return token_adv, token_adv


def compute_custom_advantage(data: Any, adv_estimator: Any, config: Any) -> Any:
    name = getattr(adv_estimator, "value", adv_estimator)
    name = str(name)
    if not _CAST_ADV_ENTRY_LOGGED[0]:
        _wrong = _cfg_get(config, "algorithm.rlsd.custom_adv_estimator")
        _right = _cfg_get(config, "rlsd.custom_adv_estimator")
        _msg = (
            f"[CAST advantage-entry] adv_estimator={name!r} "
            f"wrong_path(algorithm.rlsd.*)={_wrong!r} "
            f"right_path(rlsd.*)={_right!r}"
        )
        logger.info(_msg)
        import sys
        print(_msg, file=sys.stderr, flush=True)
        _CAST_ADV_ENTRY_LOGGED[0] = True
    fn = get_adv_estimator_fn(name)
    batch = getattr(data, "batch", None)
    non_tensor_batch = getattr(data, "non_tensor_batch", {}) or {}
    token_level_rewards = _batch_get(batch, "token_level_rewards", "token_level_scores")
    response_mask = _batch_get(batch, "response_mask", "attention_mask")
    if response_mask is None:
        responses = _batch_get(batch, "responses")
        response_mask = torch.ones_like(responses, dtype=torch.float32)
    response_mask = _as_tensor(response_mask).float()
    token_level_rewards = _as_tensor(token_level_rewards, response_mask).float()
    if token_level_rewards.shape[-1] != response_mask.shape[-1]:
        token_level_rewards = token_level_rewards[..., -response_mask.shape[-1] :]
    index = non_tensor_batch.get("uid", non_tensor_batch.get("index", None))
    global_step = None
    try:
        from .teacher_ema import _extract_global_step_from_data, _get_trainer_global_step

        global_step = _extract_global_step_from_data(data)
        if global_step is None:
            global_step = _get_trainer_global_step(0)
    except Exception:
        global_step = 0
    adv_kwargs = {
        "batch": batch,
        "non_tensor_batch": non_tensor_batch,
        "old_log_probs": _batch_get(batch, "old_log_probs", "old_log_prob"),
        "ref_log_prob": _batch_get(batch, "ref_log_prob", "ref_log_probs"),
        "teacher_log_probs": _batch_get(batch, "teacher_log_probs", "teacher_logprobs", "teacher_log_probs_all"),
        "global_step": global_step,
    }
    advantages, returns = fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config=config,
        **adv_kwargs,
    )
    meta = getattr(data, "meta_info", None)
    if meta is None:
        data.meta_info = {"cast_adv_estimator": name, "cast_adv_branch": "custom"}
        meta = data.meta_info
    elif isinstance(meta, dict):
        meta["cast_adv_estimator"] = name
        meta["cast_adv_branch"] = "custom"
    if isinstance(meta, dict) and _LATEST_CAST_TRAIN_METRICS:
        metrics_bucket = meta.setdefault("metrics", {})
        if isinstance(metrics_bucket, dict):
            metrics_bucket.update(_LATEST_CAST_TRAIN_METRICS)
        meta["cast_metrics"] = dict(_LATEST_CAST_TRAIN_METRICS)
    try:
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    except Exception:
        batch["advantages"] = advantages
        batch["returns"] = returns
    return data
