"""Fix async vLLM rollout using base weights instead of the synced actor LoRA.

Root cause:
  FSDP->vLLM weight sync calls add_lora(...), but ChatCompletionScheduler requests
  model=<base_name>. vLLM OpenAI serving then sets lora_request=None, so generation
  ignores the loaded adapter.

Fix:
  1) After each AsyncvLLMServer.wake_up (which triggers worker weight sync), refresh
     OpenAIServingModels.lora_requests from engine.list_loras().
  2) When a chat request names the base model, auto-select that actor LoRA request.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_ACTOR_LORA_PREFIX = "__cast_actor_"
_PATCHED_ADAPTERS = False
_PATCHED_WAKE_UP = False


def _make_stub_lora_request(lora_int_id: int) -> Any:
    from vllm.lora.request import LoRARequest

    return LoRARequest(
        lora_name=f"{_ACTOR_LORA_PREFIX}{lora_int_id}",
        lora_int_id=int(lora_int_id),
        lora_path="/simon-stub-path",
    )


def _actor_lora_from_registry(models: Any) -> Any | None:
    reqs = list(getattr(models, "lora_requests", []) or [])
    actor = [r for r in reqs if str(getattr(r, "lora_name", "")).startswith(_ACTOR_LORA_PREFIX)]
    if actor:
        return max(actor, key=lambda r: int(getattr(r, "lora_int_id", 0) or 0))
    # Fallback: FSDP sync names adapters as str(lora_int_id).
    numeric = [r for r in reqs if str(getattr(r, "lora_name", "")).isdigit()]
    if numeric:
        return max(numeric, key=lambda r: int(getattr(r, "lora_int_id", 0) or 0))
    return None


async def refresh_actor_lora_registry(server: Any) -> None:
    """Populate OpenAI serving registry from engine-loaded LoRA ids after sync."""
    try:
        from verl_rlsd.merge_lora_sync import merge_lora_sync_enabled

        if merge_lora_sync_enabled():
            return
    except Exception:
        pass
    engine = getattr(server, "engine", None)
    serving = getattr(server, "openai_serving_chat", None)
    if engine is None or serving is None:
        return
    models = getattr(serving, "models", None)
    if models is None:
        return

    list_loras = getattr(engine, "list_loras", None)
    if list_loras is None:
        return

    try:
        ids = await list_loras()
    except Exception as exc:
        logger.warning("CAST async LoRA fix: list_loras failed: %s", exc)
        return

    ids = sorted(int(x) for x in (ids or []))
    # Keep non-actor entries (e.g. teacher_ema loaded via HTTP).
    others = [
        r
        for r in list(getattr(models, "lora_requests", []) or [])
        if not str(getattr(r, "lora_name", "")).startswith(_ACTOR_LORA_PREFIX)
    ]
    if not ids:
        models.lora_requests = others
        logger.info("CAST async LoRA fix: no engine LoRAs after wake_up")
        return

    actor_req = _make_stub_lora_request(ids[-1])
    models.lora_requests = others + [actor_req]
    logger.info(
        "CAST async LoRA fix: registered actor LoRA name=%s id=%s (engine_ids=%s)",
        actor_req.lora_name,
        actor_req.lora_int_id,
        ids,
    )
    print(
        f"[CAST] async LoRA registry: actor={actor_req.lora_name} "
        f"id={actor_req.lora_int_id} engine_ids={ids}",
        flush=True,
    )


def patch_openai_serving_auto_actor_lora() -> None:
    """Base-model chat requests activate the synced actor LoRA."""
    global _PATCHED_ADAPTERS
    # Dense merge mode samples without adapters; do not force lora_request.
    try:
        from verl_rlsd.merge_lora_sync import merge_lora_sync_enabled

        if merge_lora_sync_enabled():
            print("[CAST] skip async LoRA activation patch (merge-LoRA mode)", flush=True)
            _PATCHED_ADAPTERS = True
            return
    except Exception:
        pass
    from vllm.entrypoints.openai.serving_engine import OpenAIServing

    if getattr(OpenAIServing._maybe_get_adapters, "_cast_async_lora_fixed", False):
        _PATCHED_ADAPTERS = True
        return

    original = OpenAIServing._maybe_get_adapters

    def _patched_maybe_get_adapters(self, request):  # type: ignore[no-untyped-def]
        # Base-model requests: still activate the synced actor LoRA for RL rollout.
        if self._is_model_supported(request.model):
            actor = _actor_lora_from_registry(self.models)
            if actor is not None:
                return actor, None
            return None, None
        return original(self, request)

    _patched_maybe_get_adapters._cast_async_lora_fixed = True  # type: ignore[attr-defined]
    OpenAIServing._maybe_get_adapters = _patched_maybe_get_adapters  # type: ignore[method-assign]
    _PATCHED_ADAPTERS = True
    logger.info("CAST async LoRA fix: patched OpenAIServing._maybe_get_adapters")


def _unwrap_ray_actor_class(cls: Any) -> Any:
    cur = cls
    for _ in range(4):
        inner = getattr(cur, "__ray_actor_class__", None) or getattr(cur, "__wrapped__", None)
        if inner is None:
            break
        cur = inner
    return cur


def patch_async_vllm_server_wake_up() -> None:
    """Refresh LoRA registry after AsyncvLLMServer.wake_up (fallback if on-disk inject missing)."""
    global _PATCHED_WAKE_UP
    try:
        from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer
    except Exception as exc:
        logger.warning("CAST async LoRA fix: cannot import AsyncvLLMServer: %s", exc)
        return

    server_cls = _unwrap_ray_actor_class(AsyncvLLMServer)
    wake_up = getattr(server_cls, "wake_up", None)
    if wake_up is None or getattr(wake_up, "_cast_async_lora_fixed", False):
        _PATCHED_WAKE_UP = True
        return

    async def _patched_wake_up(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = await wake_up(self, *args, **kwargs)
        try:
            await refresh_actor_lora_registry(self)
        except Exception as exc:
            logger.warning("CAST async LoRA fix: refresh after wake_up failed: %s", exc)
            print(f"[CAST] async LoRA registry refresh failed: {exc}", flush=True)
        return result

    _patched_wake_up._cast_async_lora_fixed = True  # type: ignore[attr-defined]
    server_cls.wake_up = _patched_wake_up  # type: ignore[method-assign]
    _PATCHED_WAKE_UP = True
    logger.info("CAST async LoRA fix: patched AsyncvLLMServer.wake_up")


def ensure_async_lora_activation_patches() -> None:
    """Idempotent patches for async vLLM LoRA activation."""
    patch_openai_serving_auto_actor_lora()
    patch_async_vllm_server_wake_up()
    print("[CAST] async LoRA activation fix applied", flush=True)
