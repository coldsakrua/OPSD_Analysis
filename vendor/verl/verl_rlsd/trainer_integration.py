"""Runtime integration with veRL RayPPOTrainer advantage computation."""

from __future__ import annotations

import functools
import importlib
import sys
from typing import Any

from .advantage import compute_custom_advantage, get_custom_adv_estimator_name, is_registered_adv_estimator

_ORIGINAL_COMPUTE_ADVANTAGE = None
_PATCHED = False
_ROUTING_LOGGED = False


def cast_compute_advantage(data: Any, adv_estimator: Any, *args: Any, **kwargs: Any) -> Any:
    """Route to CAST custom advantage when configured, else native veRL."""
    global _ROUTING_LOGGED
    config = kwargs.get("config")
    custom_name = get_custom_adv_estimator_name(config)
    if custom_name and is_registered_adv_estimator(custom_name):
        if not _ROUTING_LOGGED:
            print(
                f"[CAST advantage-route] using custom estimator={custom_name!r} (not GRPO fallback)",
                file=sys.stderr,
                flush=True,
            )
            _ROUTING_LOGGED = True
        return compute_custom_advantage(data, custom_name, config)
    if custom_name:
        print(
            f"[CAST advantage-route] WARNING: custom_adv_estimator={custom_name!r} "
            f"is not registered; falling back to native adv_estimator={adv_estimator!r} (GRPO path)",
            file=sys.stderr,
            flush=True,
        )
    elif not _ROUTING_LOGGED:
        print(
            f"[CAST advantage-route] no custom_adv_estimator configured; using native {adv_estimator!r}",
            file=sys.stderr,
            flush=True,
        )
        _ROUTING_LOGGED = True
    if _ORIGINAL_COMPUTE_ADVANTAGE is None:
        raise RuntimeError("CAST advantage routing is active but original compute_advantage is missing.")
    return _ORIGINAL_COMPUTE_ADVANTAGE(data, adv_estimator, *args, **kwargs)


def patch_advantage_routing() -> None:
    global _ORIGINAL_COMPUTE_ADVANTAGE, _PATCHED
    if _PATCHED:
        return
    try:
        ray_trainer_mod = importlib.import_module("verl.trainer.ppo.ray_trainer")
    except Exception:
        return
    original = getattr(ray_trainer_mod, "compute_advantage", None)
    if original is None or getattr(original, "_cast_patched", False):
        _PATCHED = True
        return
    _ORIGINAL_COMPUTE_ADVANTAGE = original

    @functools.wraps(original)
    def wrapped(data, adv_estimator, *args, **kwargs):  # type: ignore[no-untyped-def]
        return cast_compute_advantage(data, adv_estimator, *args, **kwargs)

    wrapped._cast_patched = True
    ray_trainer_mod.compute_advantage = wrapped
    _PATCHED = True
