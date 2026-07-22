"""Merge LoRA into dense weights before vLLM sync (prefer generate speed).

When RLSD_MERGE_LORA_FOR_ASYNC_VLLM / actor_rollout_ref.model.lora.merge is true,
skip TensorLoRARequest+add_lora and instead:

  1) temporarily merge adapters on the FSDP/Peft module (writeback=False)
  2) export dense base weights
  3) unmerge (training LoRA state unchanged)
  4) vLLM model.load_weights(...)  → sample on the fast dense path

This matches the intended meaning of lora.merge=true in CAST configs.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED = False


def merge_lora_sync_enabled() -> bool:
    for key in ("RLSD_MERGE_LORA_FOR_ASYNC_VLLM", "CAST_MERGE_LORA_FOR_ASYNC_VLLM"):
        val = os.environ.get(key, "")
        if str(val).strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _cpu_copy(param: Any) -> Any:
    import torch

    t = param
    if hasattr(t, "full_tensor"):
        try:
            t = t.full_tensor()
        except Exception:
            pass
    if isinstance(t, torch.Tensor):
        return t.detach().contiguous().cpu()
    return t


def collect_merged_dense_params(module: Any) -> OrderedDict:
    """Export merged dense weights without permanently altering the train LoRA."""
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

    from verl.utils.fsdp_utils import fsdp_version

    peft_model = getattr(module, "_fsdp_wrapped_module", module)
    tuner = getattr(peft_model, "base_model", None)
    merge_fn = None
    unmerge_fn = None
    if tuner is not None and hasattr(tuner, "merge_adapter"):
        merge_fn = tuner.merge_adapter
        unmerge_fn = getattr(tuner, "unmerge_adapter", None)
    elif hasattr(peft_model, "merge_adapter"):
        merge_fn = peft_model.merge_adapter
        unmerge_fn = getattr(peft_model, "unmerge_adapter", None)

    params: OrderedDict[str, Any] = OrderedDict()

    def _export_from_peft() -> None:
        base = getattr(getattr(peft_model, "base_model", peft_model), "model", peft_model)
        for name, param in base.state_dict().items():
            if any(x in name for x in ("_flat_param", "lora_")):
                continue
            name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
            params[name] = _cpu_copy(param)

    merged = False
    try:
        if fsdp_version(module) > 0:
            with FSDP.summon_full_params(module, writeback=False):
                if merge_fn is not None:
                    merge_fn()
                    merged = True
                _export_from_peft()
                if merged and unmerge_fn is not None:
                    unmerge_fn()
                    merged = False
        else:
            if merge_fn is not None:
                merge_fn()
                merged = True
            _export_from_peft()
    finally:
        if merged and unmerge_fn is not None:
            try:
                unmerge_fn()
            except Exception as exc:
                logger.warning("CAST merge sync: unmerge_adapter failed: %s", exc)

    return params


def _patched_update_params(self, updated_params, peft_config=None):  # type: ignore[no-untyped-def]
    """Replace add_lora branch with merged dense load_weights when merge is on."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # torch<2.5
        from torch.distributed._tensor import DTensor

    from verl.utils.device import get_device_id
    from verl.utils.vllm_utils import patch_vllm_moe_model_weight_loader

    model = self.model_runner.model

    if merge_lora_sync_enabled() and peft_config is not None:
        merged = collect_merged_dense_params(self.module)
        print(
            f"[CAST] merge-LoRA sync: exporting {len(merged)} dense tensors "
            f"(skip add_lora; base_sync_done={getattr(self, 'base_sync_done', None)})",
            flush=True,
        )
        updated_params = merged
        peft_config = None

    if peft_config:
        if self.base_sync_done:
            # Original adapter path (merge disabled).
            import time
            from dataclasses import asdict

            from verl.utils.vllm_utils import TensorLoRARequest

            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_request = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=updated_params,
            )
            from verl_rlsd.launch import _rlsd_add_lora

            _rlsd_add_lora(self, lora_request)
            logger.info("vLLM load weights, loaded_params: %s", len(updated_params))
            return

        def replace_lora_wrapper(k: str) -> str:
            stacked_params = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            if any(k.endswith(f"{s}.weight") for s in stacked_params):
                return k.replace(".weight", ".base_layer.weight")
            if any(k.endswith(f"{s}.bias") for s in stacked_params):
                return k.replace(".bias", ".base_layer.bias")
            return k

        updated_params = {replace_lora_wrapper(k): v for k, v in updated_params.items()}

    patch_vllm_moe_model_weight_loader(model)
    device = get_device_id()
    loaded_params = model.load_weights(
        (
            (
                name,
                param.to(device, non_blocking=True).full_tensor()
                if isinstance(param, DTensor)
                else param,
            )
            for name, param in updated_params.items()
        )
    )
    self.base_sync_done = True
    n_loaded = len(loaded_params) if loaded_params else -1
    logger.info("vLLM load weights, loaded_params: %s", n_loaded)
    print(f"[CAST] vLLM dense load_weights done: loaded_params={n_loaded}", flush=True)


def patch_fsdp_vllm_merge_lora_sync() -> None:
    """Idempotent: make FSDPVLLMShardingManager.update_params honor merge=true."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
    except Exception as exc:
        print(f"[CAST] merge-LoRA sync patch skipped: cannot import manager: {exc}", flush=True)
        return

    if getattr(FSDPVLLMShardingManager.update_params, "_cast_merge_lora_sync", False):
        _PATCHED = True
        return

    FSDPVLLMShardingManager.update_params = _patched_update_params  # type: ignore[method-assign]
    FSDPVLLMShardingManager.update_params._cast_merge_lora_sync = True  # type: ignore[attr-defined]
    _PATCHED = True
    enabled = merge_lora_sync_enabled()
    print(
        f"[CAST] merge-LoRA sync patch applied (enabled={enabled})",
        flush=True,
    )
