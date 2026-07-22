"""Ensure CAST runtime patches are applied inside Ray TaskRunner workers."""

from __future__ import annotations

from .async_lora_fix import ensure_async_lora_activation_patches
from .merge_lora_sync import merge_lora_sync_enabled, patch_fsdp_vllm_merge_lora_sync
from .rollout_metrics import patch_rollout_length_logging
from .teacher_ema import patch_teacher_ema
from .trainer_integration import patch_advantage_routing


def ensure_cast_worker_patches() -> None:
    """Apply CAST patches inside Ray GPU worker processes (idempotent)."""
    from .teacher_ema import patch_cast_worker

    patch_cast_worker()
    # Sharding-manager runs in WorkerDict: merge-then-load_weights must be patched here.
    patch_fsdp_vllm_merge_lora_sync()
    print("[CAST worker bootstrap] internal teacher + merge-LoRA sync patches applied", flush=True)


def ensure_cast_patches() -> None:
    """Idempotent patch entry used by veRL TaskRunner before trainer init."""
    patch_advantage_routing()
    patch_teacher_ema()
    # Must run inside TaskRunner: driver-side metric patches do not affect this process,
    # so cast/* never reached wandb when only launch.py patched compute_data_metrics.
    patch_rollout_length_logging()
    patch_fsdp_vllm_merge_lora_sync()
    # Adapter auto-activation only needed when NOT merging into dense weights.
    if not merge_lora_sync_enabled():
        ensure_async_lora_activation_patches()
    print(
        "[CAST bootstrap] advantage routing + trainer-side teacher + cast metrics + "
        f"merge_lora={merge_lora_sync_enabled()} patches applied",
        flush=True,
    )
