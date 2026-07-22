from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Any

from . import advantage as custom_advantage
from .qwen3_chat_template import (
    install_qwen3_no_think_chat_template,
    strip_empty_thinking_enabled,
    strip_empty_thinking_generation_prompt,
)
from .olmo_chat_template import maybe_install_olmo_chat_template
from .batch_truncate import prepare_rollout_batch
from .rollout_dump import patch_rollout_dump
from .rollout_metrics import patch_rollout_length_logging
from .teacher_ema import patch_teacher_ema
from .trainer_integration import patch_advantage_routing


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


_COMPUTE_ADVANTAGE_CANONICAL = (
    "                        batch = compute_advantage(\n"
    "                            batch,\n"
    "                            adv_estimator=self.config.algorithm.adv_estimator,\n"
    "                            gamma=self.config.algorithm.gamma,\n"
    "                            lam=self.config.algorithm.lam,\n"
    "                            num_repeat=self.config.actor_rollout_ref.rollout.n,\n"
    "                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,\n"
    "                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,\n"
    "                            config=self.config.algorithm,\n"
    "                        )"
)
_COMPUTE_ADVANTAGE_MARKER = "# RLSD_CUSTOM_ADVANTAGE_PATCH"


def _restore_ray_trainer_compute_advantage_on_disk() -> None:
    """Remove broken CAST disk patches and restore canonical veRL compute_advantage call."""
    try:
        ray_trainer_mod = importlib.import_module("verl.trainer.ppo.ray_trainer")
    except Exception as exc:
        print(f"[CAST restore] skip ray_trainer restore: {exc}", flush=True)
        return
    path = getattr(ray_trainer_mod, "__file__", None)
    if not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    original = text
    if _COMPUTE_ADVANTAGE_MARKER in text:
        start = text.index(_COMPUTE_ADVANTAGE_MARKER)
        end = text.find("\n                    # update critic", start)
        if end == -1:
            end = text.find("\n                    if self.use_critic:", start)
        if end != -1:
            text = text[:start] + _COMPUTE_ADVANTAGE_CANONICAL + text[end:]
            print("[CAST restore] removed legacy disk advantage patch from ray_trainer.py", flush=True)
    broken_prefix = "                                                                                                                                                                        batch = compute_advantage("
    if broken_prefix in text:
        text = text.replace(broken_prefix, "                        batch = compute_advantage(", 1)
        print("[CAST restore] fixed corrupted compute_advantage indentation", flush=True)
    if text != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    try:
        importlib.reload(ray_trainer_mod)
    except Exception:
        pass


def _patch_compute_advantage() -> None:
    """Restore veRL ray_trainer and route advantages through in-process CAST integration."""
    _restore_ray_trainer_compute_advantage_on_disk()
    patch_advantage_routing()
    print("[CAST] advantage routing patched in-process (TaskRunner uses bootstrap.ensure_cast_patches)", flush=True)


def _log_rollout_batch_prep(mode: str, count: int) -> None:
    if count <= 0:
        return
    if mode == "discard":
        print(
            f"[rlsd] discarded {count} truncated rollout sample(s) "
            f"(max_response_len={os.environ.get('RLSD_MAX_RESPONSE_LENGTH', 'unset')})",
            flush=True,
        )
        return
    print(
        f"[rlsd] truncated rollout batch by {count} tokens "
        f"(max_seq_len={os.environ.get('RLSD_MAX_SEQ_LEN', 'unset')}, "
        f"max_response_len={os.environ.get('RLSD_MAX_RESPONSE_LENGTH', 'unset')})",
        flush=True,
    )


def _patch_truncate_rollout_batch() -> None:
    try:
        ray_trainer = importlib.import_module("verl.trainer.ppo.ray_trainer")
    except Exception:
        return
    original = getattr(ray_trainer, "compute_response_mask", None)
    if original is None or getattr(original, "_rlsd_truncate_patched", False):
        return

    def patched_compute_response_mask(data: Any):
        mode, count = prepare_rollout_batch(data)
        _log_rollout_batch_prep(mode, count)
        return original(data)

    patched_compute_response_mask._rlsd_truncate_patched = True
    ray_trainer.compute_response_mask = patched_compute_response_mask


def _patch_truncate_before_logprob() -> None:
    try:
        fsdp_workers = importlib.import_module("verl.workers.fsdp_workers")
    except Exception:
        return
    worker_cls = getattr(fsdp_workers, "ActorRolloutRefWorker", None)
    if worker_cls is None:
        return
    original = getattr(worker_cls, "compute_log_prob", None)
    if original is None or getattr(original, "_rlsd_truncate_patched", False):
        return

    def patched_compute_log_prob(self, data: Any):
        mode, count = prepare_rollout_batch(data)
        _log_rollout_batch_prep(mode, count)
        return original(self, data)

    patched_compute_log_prob._rlsd_truncate_patched = True
    worker_cls.compute_log_prob = patched_compute_log_prob


_RAY_INIT_OLD = "ray.init(namespace=namespace)"
_RAY_INIT_NEW = (
    'ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), '
    "namespace=namespace, ignore_reinit_error=True)"
)
_RLSD_STRIP_EMPTY_THINKING_PATCH = "# RLSD_STRIP_EMPTY_THINKING_PATCH"
_RLSD_INSTALL_NO_THINK_CHAT_TEMPLATE_PATCH = "# RLSD_INSTALL_NO_THINK_CHAT_TEMPLATE_PATCH"
_RLSD_VLLM_CHAT_TEMPLATE_PATCH = "# RLSD_VLLM_CHAT_TEMPLATE_PATCH"
_APPLY_CHAT_NO_THINK_OLD = """    def _apply_chat_no_think(messages, *args, **kwargs):
        kw = dict(kwargs)
        kw["enable_thinking"] = False
        try:
            return _orig_apply_chat(messages, *args, **kw)
        except TypeError:
            kw.pop("enable_thinking", None)
            return _orig_apply_chat(messages, *args, **kw)"""
_APPLY_CHAT_NO_THINK_NEW = """    def _apply_chat_no_think(messages, *args, **kwargs):
        kw = dict(kwargs)
        kw["enable_thinking"] = False
        try:
            out = _orig_apply_chat(messages, *args, **kw)
        except TypeError:
            kw.pop("enable_thinking", None)
            out = _orig_apply_chat(messages, *args, **kw)
        if (
            isinstance(out, str)
            and _rlsd_os.environ.get("STRIP_EMPTY_THINKING_GENERATION_PROMPT", "false").strip().lower()
            in {"1", "true", "yes", "on"}
            and kw.get("add_generation_prompt", True)
        ):
            try:
                from verl_rlsd.qwen3_chat_template import strip_empty_thinking_generation_prompt
                out = strip_empty_thinking_generation_prompt(out)
            except Exception:
                pass
        return out"""


_ASYNC_VLLM_ENGINE_KWARGS_MARKER = "# RLSD_ASYNC_VLLM_ENGINE_KWARGS_PATCH"
_ASYNC_VLLM_ENGINE_KWARGS_OLD = """        print(f"override_generation_config: {kwargs}")

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=ExternalRayDistributedExecutor if os.environ.get("VERL_VLLM_USE_RAY_BACKEND", "1") == "1" else None,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format="auto",
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=self.vllm_dp_rank,
        )"""
_ASYNC_VLLM_ENGINE_KWARGS_NEW = """        print(f"override_generation_config: {kwargs}")

        from copy import deepcopy

        from omegaconf import OmegaConf

        engine_kwargs = {}
        if getattr(config, "engine_kwargs", None) is not None and getattr(
            config.engine_kwargs, "vllm", None
        ) is not None:
            engine_kwargs = OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        model_config = self.config.model
        lora_rank = int(getattr(model_config, "lora_rank", 0) or 0)
        _merge_lora = str(__import__("os").environ.get("RLSD_MERGE_LORA_FOR_ASYNC_VLLM", "")).strip().lower() in {
            "1", "true", "yes", "on",
        }
        if _merge_lora:
            # Dense merged sync: keep vLLM on the fast non-adapter generate path.
            engine_kwargs["enable_lora"] = False
            engine_kwargs.pop("max_loras", None)
            engine_kwargs.pop("max_lora_rank", None)
            print("[CAST] merge-LoRA mode: vLLM enable_lora=false", flush=True)
        elif lora_rank > 0 and not engine_kwargs.get("enable_lora"):
            engine_kwargs["enable_lora"] = True
            engine_kwargs.setdefault("max_lora_rank", lora_rank)
            engine_kwargs.setdefault("max_loras", max(int(engine_kwargs.get("max_loras", 0) or 0), 1))
        """ + _ASYNC_VLLM_ENGINE_KWARGS_MARKER + """

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=True,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=ExternalRayDistributedExecutor if os.environ.get("VERL_VLLM_USE_RAY_BACKEND", "1") == "1" else None,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format="auto",
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=self.vllm_dp_rank,
            **engine_kwargs,
        )"""


_CAST_ASYNC_LORA_FIX_MARKER = "# CAST_ASYNC_LORA_ACTIVATION_FIX"
_CAST_ASYNC_LORA_WAKE_UP_OLD = """    async def wake_up(self):
        await self.engine.wake_up()
"""
_CAST_ASYNC_LORA_WAKE_UP_NEW = """    async def wake_up(self):
        await self.engine.wake_up()
        # CAST_ASYNC_LORA_ACTIVATION_FIX
        try:
            from verl_rlsd.merge_lora_sync import merge_lora_sync_enabled
            if merge_lora_sync_enabled():
                # Dense merge path: no LoRA manager / list_loras.
                return
            from verl_rlsd.async_lora_fix import refresh_actor_lora_registry
            await refresh_actor_lora_registry(self)
        except Exception as _cast_lora_exc:
            print(f"[CAST] async LoRA registry refresh failed: {_cast_lora_exc}", flush=True)
"""
_CAST_ASYNC_LORA_WAKE_UP_LEGACY = """    async def wake_up(self):
        await self.engine.wake_up()
        # CAST_ASYNC_LORA_ACTIVATION_FIX
        try:
            from verl_rlsd.async_lora_fix import refresh_actor_lora_registry
            await refresh_actor_lora_registry(self)
        except Exception as _cast_lora_exc:
            print(f"[CAST] async LoRA registry refresh failed: {_cast_lora_exc}", flush=True)
"""
_CAST_ASYNC_LORA_EOF_SNIPPET = """
# CAST_ASYNC_LORA_ACTIVATION_FIX
try:
    from verl_rlsd.async_lora_fix import patch_openai_serving_auto_actor_lora
    patch_openai_serving_auto_actor_lora()
except Exception as _cast_async_lora_exc:
    print(f"[CAST] async LoRA activation fix failed: {_cast_async_lora_exc}", flush=True)
"""


_CAST_MERGE_DISABLE_LORA_OLD = """        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        model_config = self.config.model
        lora_rank = int(getattr(model_config, "lora_rank", 0) or 0)
        if lora_rank > 0 and not engine_kwargs.get("enable_lora"):
            engine_kwargs["enable_lora"] = True
            engine_kwargs.setdefault("max_lora_rank", lora_rank)
            engine_kwargs.setdefault("max_loras", max(int(engine_kwargs.get("max_loras", 0) or 0), 1))
        # RLSD_ASYNC_VLLM_ENGINE_KWARGS_PATCH
"""
_CAST_MERGE_DISABLE_LORA_NEW = """        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        model_config = self.config.model
        lora_rank = int(getattr(model_config, "lora_rank", 0) or 0)
        _merge_lora = str(os.environ.get("RLSD_MERGE_LORA_FOR_ASYNC_VLLM", "")).strip().lower() in {
            "1", "true", "yes", "on",
        }
        if _merge_lora:
            # Dense merged sync: keep vLLM on the fast non-adapter generate path.
            engine_kwargs["enable_lora"] = False
            engine_kwargs.pop("max_loras", None)
            engine_kwargs.pop("max_lora_rank", None)
            print("[CAST] merge-LoRA mode: vLLM enable_lora=false", flush=True)
        elif lora_rank > 0 and not engine_kwargs.get("enable_lora"):
            engine_kwargs["enable_lora"] = True
            engine_kwargs.setdefault("max_lora_rank", lora_rank)
            engine_kwargs.setdefault("max_loras", max(int(engine_kwargs.get("max_loras", 0) or 0), 1))
        # RLSD_ASYNC_VLLM_ENGINE_KWARGS_PATCH
        # CAST_MERGE_DISABLE_LORA_PATCH
"""


def _patch_verl_vllm_async_server_on_disk() -> None:
    """Patch installed verl async vLLM server for Ray cluster + LoRA engine kwargs."""
    try:
        mod = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_async_server")
    except Exception:
        return
    path = getattr(mod, "__file__", None)
    if not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    changed = False
    if _RAY_INIT_OLD in text and _RAY_INIT_NEW not in text:
        text = text.replace(_RAY_INIT_OLD, _RAY_INIT_NEW, 1)
        changed = True
    if _ASYNC_VLLM_ENGINE_KWARGS_MARKER not in text:
        if _ASYNC_VLLM_ENGINE_KWARGS_OLD in text:
            text = text.replace(_ASYNC_VLLM_ENGINE_KWARGS_OLD, _ASYNC_VLLM_ENGINE_KWARGS_NEW, 1)
            changed = True
    if "CAST_MERGE_DISABLE_LORA_PATCH" not in text and _CAST_MERGE_DISABLE_LORA_OLD in text:
        text = text.replace(_CAST_MERGE_DISABLE_LORA_OLD, _CAST_MERGE_DISABLE_LORA_NEW, 1)
        changed = True
        print(f"[CAST] installed merge-LoRA enable_lora=false gate in {path}", flush=True)
    # Ensure chat uses synced actor LoRA when request.model is the base name.
    if "refresh_actor_lora_registry" not in text and _CAST_ASYNC_LORA_WAKE_UP_OLD in text:
        text = text.replace(_CAST_ASYNC_LORA_WAKE_UP_OLD, _CAST_ASYNC_LORA_WAKE_UP_NEW, 1)
        changed = True
    # Upgrade legacy wake_up inject to skip list_loras under merge mode.
    if _CAST_ASYNC_LORA_WAKE_UP_LEGACY in text and "Dense merge path: no LoRA manager" not in text:
        text = text.replace(_CAST_ASYNC_LORA_WAKE_UP_LEGACY, _CAST_ASYNC_LORA_WAKE_UP_NEW, 1)
        changed = True
        print(f"[CAST] upgraded async LoRA wake_up merge-guard in {path}", flush=True)
    if "patch_openai_serving_auto_actor_lora" not in text:
        text = text.rstrip() + "\n" + _CAST_ASYNC_LORA_EOF_SNIPPET
        changed = True
        print(f"[CAST] installed async LoRA activation fix in {path}", flush=True)
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


_FSDP_VLLM_ADD_LORA_OLD = "self.inference_engine.llm_engine.add_lora(lora_reqest)"
_FSDP_VLLM_ADD_LORA_CALL = (
    '__import__("verl_rlsd.launch", fromlist=["_rlsd_add_lora"]).'
    '_rlsd_add_lora(self, lora_reqest)  # RLSD_ASYNC_VLLM_ADD_LORA_PATCH'
)


def _rlsd_lora_tensor_fingerprint(lora_request: Any) -> str:
    """Stable short fingerprint of TensorLoRARequest weights for sync diagnostics.

    Important: only touch a few elements per tensor on-device. Never `.cpu()` the
    full parameter — that dominated step time in the first fixed withgt run.
    """
    import hashlib

    tensors = getattr(lora_request, "lora_tensors", None)
    if not isinstance(tensors, dict) or not tensors:
        return "no_tensors"
    h = hashlib.sha1()
    # Cap work: hash key list + a few tensors fully-sampled lightly.
    keys = sorted(tensors.keys())
    h.update(f"n={len(keys)}".encode("utf-8"))
    for key in keys[:: max(1, len(keys) // 16)][:16]:
        h.update(str(key).encode("utf-8"))
        val = tensors[key]
        try:
            import torch

            if isinstance(val, torch.Tensor):
                t = val.detach()
                if hasattr(t, "full_tensor"):
                    try:
                        t = t.full_tensor()
                    except Exception:
                        pass
                h.update(str(tuple(t.shape)).encode("utf-8"))
                flat = t.reshape(-1)
                n = int(flat.numel())
                if n == 0:
                    continue
                idx = torch.tensor([0, n // 2, n - 1] if n >= 3 else list(range(n)), device=flat.device)
                sample = flat.index_select(0, idx).float().cpu().numpy().tobytes()
                h.update(sample)
                continue
        except Exception:
            pass
        h.update(repr(val)[:64].encode("utf-8", errors="ignore"))
    return h.hexdigest()[:12]


def _rlsd_add_lora(sharding_manager: Any, lora_request: Any) -> None:
    """Compatibility wrapper for veRL LoRA sync across vLLM v0/v1 objects."""
    import asyncio
    import inspect

    lora_name = getattr(lora_request, "lora_name", None)
    lora_int_id = getattr(lora_request, "lora_int_id", None)
    n_tensors = len(getattr(lora_request, "lora_tensors", None) or {})
    fp = _rlsd_lora_tensor_fingerprint(lora_request)
    print(
        f"[CAST] vLLM add_lora sync: name={lora_name} id={lora_int_id} "
        f"n_tensors={n_tensors} fp={fp}",
        flush=True,
    )

    inference_engine = getattr(sharding_manager, "inference_engine", None)
    candidates = [
        getattr(inference_engine, "llm_engine", None),
        inference_engine,
        getattr(inference_engine, "worker", None),
        getattr(getattr(inference_engine, "worker", None), "model_runner", None),
        getattr(sharding_manager, "model_runner", None),
    ]
    seen: set[int] = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        add_lora = getattr(candidate, "add_lora", None)
        if add_lora is None:
            continue
        try:
            result = add_lora(lora_request)
        except AttributeError as exc:
            if "lora_manager" in str(exc):
                last_error = exc
                continue
            raise
        if inspect.isawaitable(result):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None
            if loop is None:
                asyncio.run(result)
            elif loop.is_running():
                asyncio.run_coroutine_threadsafe(result, loop).result()
            else:
                loop.run_until_complete(result)
        print(
            f"[CAST] vLLM add_lora OK via {type(candidate).__name__}: "
            f"id={lora_int_id} fp={fp}",
            flush=True,
        )
        return
    if last_error is not None:
        raise AttributeError(
            "vLLM exposes add_lora but was started without a LoRA manager. "
            "Enable actor_rollout_ref.rollout.engine_kwargs.vllm.enable_lora=true "
            "and set max_loras/max_lora_rank for LoRA rollout sync. If LoRA "
            "merge is enabled, also set actor_rollout_ref.model.lora.merge=false "
            "so vLLM starts in adapter mode."
        ) from last_error
    raise AttributeError(
        "RLSD failed to find a vLLM add_lora API; this veRL/vLLM combination "
        "does not expose llm_engine.add_lora or an equivalent add_lora method."
    )


def _find_module_file(module_name: str, relative_path: str) -> str | None:
    try:
        spec = importlib.util.find_spec(module_name)
        path = getattr(spec, "origin", None) if spec is not None else None
        if path and os.path.isfile(path):
            return path
    except Exception:
        pass
    for base in sys.path:
        if not base:
            continue
        path = os.path.join(base, relative_path)
        if os.path.isfile(path):
            return path
    return None


def _repair_bad_add_lora_patch(text: str) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if "RLSD_ASYNC_VLLM_ADD_LORA_PATCH" in line and "_rlsd_add_lora" not in line:
            start = i
            end = None
            limit = min(len(lines), i + 80)
            j = i + 1
            while j < limit:
                if "_rlsd_loop.run_until_complete(_rlsd_lora_result)" in lines[j]:
                    end = j + 1
                    break
                j += 1
            if end is None:
                out.append(line)
                i = start + 1
                continue
            indent = line[: len(line) - len(line.lstrip())]
            newline = "\n" if line.endswith("\n") else ""
            out.append(f"{indent}{_FSDP_VLLM_ADD_LORA_CALL}{newline}")
            changed = True
            i = end
            continue
        out.append(line)
        i += 1
    return "".join(out), changed


def _patch_fsdp_vllm_add_lora_on_disk() -> None:
    """Patch old veRL FSDP-vLLM sharding to support vLLM v1 LoRA sync."""
    path = _find_module_file(
        "verl.workers.sharding_manager.fsdp_vllm",
        os.path.join("verl", "workers", "sharding_manager", "fsdp_vllm.py"),
    )
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text, repaired = _repair_bad_add_lora_patch(text)
    if not repaired and "RLSD_ASYNC_VLLM_ADD_LORA_PATCH" in text:
        return
    if not repaired and _FSDP_VLLM_ADD_LORA_OLD not in text:
        return
    if not repaired:
        text = text.replace(_FSDP_VLLM_ADD_LORA_OLD, _FSDP_VLLM_ADD_LORA_CALL, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


_RLSD_APPLY_CHAT_TEMPLATE_KWARGS_MARKER = "# RLSD_APPLY_CHAT_TEMPLATE_KWARGS"
_RLSD_DISABLE_THINKING_TOKENIZER_MARKER = "# RLSD_DISABLE_THINKING_IN_CHAT_TEMPLATE_PATCH"
_RLSD_OLMO_CHAT_TEMPLATE_MARKER = "# RLSD_OLMO_CHAT_TEMPLATE_PATCH"
_RLSD_CHAT_SCHEDULER_DISABLE_THINKING_MARKER = "# RLSD_CHAT_SCHEDULER_DISABLE_THINKING_PATCH"
_RLSD_CHAT_SCHEDULER_QUIET_ROLLOUT_MARKER = "# RLSD_CHAT_SCHEDULER_QUIET_ROLLOUT_PATCH"


def _rlsd_disable_thinking_enabled() -> bool:
    raw = os.environ.get("DISABLE_THINKING_IN_CHAT_TEMPLATE", "")
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _patch_rl_dataset_apply_chat_template_kwargs_on_disk() -> None:
    """Pass data.apply_chat_template_kwargs (e.g. enable_thinking=False) in Ray workers."""
    path = _find_module_file(
        "verl.utils.dataset.rl_dataset",
        os.path.join("verl", "utils", "dataset", "rl_dataset.py"),
    )
    if not path or _RLSD_APPLY_CHAT_TEMPLATE_KWARGS_MARKER in open(path, encoding="utf-8").read():
        return

    with open(path, encoding="utf-8") as f:
        text = f.read()

    init_old = '        self.chat_template_func = config.get("chat_template_func", None)\n        self.need_tools_kwargs = config.get("need_tools_kwargs", False)'
    init_new = (
        '        self.chat_template_func = config.get("chat_template_func", None)\n'
        "        self.apply_chat_template_kwargs = dict(config.get(\"apply_chat_template_kwargs\") or {})  "
        + _RLSD_APPLY_CHAT_TEMPLATE_KWARGS_MARKER
        + "\n        self.need_tools_kwargs = config.get(\"need_tools_kwargs\", False)"
    )
    if init_old not in text:
        return
    text = text.replace(init_old, init_new, 1)

    replacements = [
        (
            "self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)",
            "self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)",
        ),
        (
            "return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))",
            "return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs))",
        ),
        (
            "self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)",
            "self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs)",
        ),
    ]
    for old, new in replacements:
        if old not in text:
            return
        text = text.replace(old, new)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _patch_chat_scheduler_disable_thinking_on_disk() -> None:
    """Pass chat_template_kwargs to vLLM async chat/completions API (rollout.mode=async)."""
    if not _rlsd_disable_thinking_enabled():
        return
    path = _find_module_file(
        "verl.workers.rollout.chat_scheduler",
        os.path.join("verl", "workers", "rollout", "chat_scheduler.py"),
    )
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if _RLSD_CHAT_SCHEDULER_DISABLE_THINKING_MARKER in text:
        return

    helper = '''
def _rlsd_chat_template_kwargs() -> dict:
    """Qwen3 non-thinking mode for async vLLM chat/completions requests."""
    import os as _rlsd_os

    if _rlsd_os.environ.get("DISABLE_THINKING_IN_CHAT_TEMPLATE", "").strip().lower() in {
        "1", "true", "yes", "y", "on",
    }:
        return {"enable_thinking": False}
    return {}


''' + _RLSD_CHAT_SCHEDULER_DISABLE_THINKING_MARKER + "\n"

    anchor = "logger = logging.getLogger(__file__)"
    if anchor not in text:
        return
    text = text.replace(anchor, anchor + "\n\n" + helper, 1)

    extra_body_old = '''    @property
    def extra_body(self) -> Dict[str, Any]:
        """Extra body pass to OpenAI API."""
        return None'''
    extra_body_new = '''    @property
    def extra_body(self) -> Dict[str, Any]:
        """Extra body pass to OpenAI API."""
        chat_kwargs = _rlsd_chat_template_kwargs()
        if chat_kwargs:
            return {"chat_template_kwargs": chat_kwargs}
        return None'''
    if extra_body_old not in text:
        return
    text = text.replace(extra_body_old, extra_body_new, 1)

    postprocess_old = (
        "        prompts = [self.tokenizer.apply_chat_template(prompt, tools=self.tool_schemas, "
        "add_generation_prompt=True, tokenize=False) for prompt in batch.non_tensor_batch[\"raw_prompt\"]]"
    )
    postprocess_new = (
        "        _rlsd_ctkw = _rlsd_chat_template_kwargs()\n"
        "        prompts = [self.tokenizer.apply_chat_template(prompt, tools=self.tool_schemas, "
        "add_generation_prompt=True, tokenize=False, **_rlsd_ctkw) "
        "for prompt in batch.non_tensor_batch[\"raw_prompt\"]]"
    )
    if postprocess_old not in text:
        return
    text = text.replace(postprocess_old, postprocess_new, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _patch_chat_scheduler_quiet_rollout_logs_on_disk() -> None:
    """Silence per-completion ToolCompletionCallback prints (very noisy for math RL)."""
    path = _find_module_file(
        "verl.workers.rollout.chat_scheduler",
        os.path.join("verl", "workers", "rollout", "chat_scheduler.py"),
    )
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if _RLSD_CHAT_SCHEDULER_QUIET_ROLLOUT_MARKER in text:
        return

    replacements = [
        (
            "        # STEP 0: check if we reach max turns\n"
            "        if self.max_assistant_turns and len(messages) >= self.max_assistant_turns:\n"
            '            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Reach max turns, done!")\n'
            "            return",
            "        # STEP 0: check if we reach max turns  "
            + _RLSD_CHAT_SCHEDULER_QUIET_ROLLOUT_MARKER
            + "\n"
            "        if self.max_assistant_turns and len(messages) >= self.max_assistant_turns:\n"
            "            return",
        ),
        (
            "        # STEP 1: check if the model called tools\n"
            '        if finish_reason != "tool_calls":\n'
            '            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] No tool called, done!")\n'
            "            return",
            '        # STEP 1: check if the model called tools\n'
            '        if finish_reason != "tool_calls":\n'
            "            return",
        ),
        (
            "        # STEP 2: call tools\n"
            "        tool_calls = completions.choices[0].message.tool_calls\n"
            '        print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Call {len(tool_calls)} tools")',
            "        # STEP 2: call tools\n"
            "        tool_calls = completions.choices[0].message.tool_calls",
        ),
        (
            "        if any(isinstance(item, Exception) for item in tool_responses):\n"
            '            print(f"[id={completions.id},turn={len(messages)},finish_reason={finish_reason}] Error when calling tools, done!")\n'
            "            return",
            "        if any(isinstance(item, Exception) for item in tool_responses):\n"
            "            return",
        ),
    ]
    for old, new in replacements:
        if old not in text:
            return
        text = text.replace(old, new, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _repair_corrupted_rlsd_tokenizer_patch() -> None:
    """Fix a broken merge that early-returned before Olmo/Qwen chat-template hooks ran."""
    path = _find_module_file("verl.utils.tokenizer", os.path.join("verl", "utils", "tokenizer.py"))
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    broken = (
        "    if tokenizer is None or not hasattr(tokenizer, \"apply_chat_template\"):\n"
        "        tokenizer = _rlsd_wrap_olmo_chat_template(tokenizer)\n"
        "    return tokenizer\n"
        "    if getattr(tokenizer.apply_chat_template, \"_rlsd_disable_thinking\", False):"
    )
    fixed = (
        "    tokenizer = _rlsd_wrap_olmo_chat_template(tokenizer)\n"
        "    if tokenizer is None or not hasattr(tokenizer, \"apply_chat_template\"):\n"
        "        return tokenizer\n"
        "    if getattr(tokenizer.apply_chat_template, \"_rlsd_disable_thinking\", False):"
    )
    if broken in text:
        text = text.replace(broken, fixed, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def _patch_hf_tokenizer_olmo_chat_template_on_disk() -> None:
    """Install Olmo ChatML template in Ray worker tokenizer loads."""
    path = _find_module_file("verl.utils.tokenizer", os.path.join("verl", "utils", "tokenizer.py"))
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if _RLSD_OLMO_CHAT_TEMPLATE_MARKER in text:
        return

    helper = '''
def _rlsd_wrap_olmo_chat_template(tokenizer):
    if tokenizer is None:
        return tokenizer
    try:
        from verl_rlsd.olmo_chat_template import maybe_install_olmo_chat_template
        maybe_install_olmo_chat_template(tokenizer)
    except Exception:
        pass
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and inner is not tokenizer:
        _rlsd_wrap_olmo_chat_template(inner)
    return tokenizer

''' + _RLSD_OLMO_CHAT_TEMPLATE_MARKER + "\n"

    anchor = "def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):"
    if anchor not in text:
        return
    text = text.replace(anchor, helper + anchor, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _patch_hf_tokenizer_disable_thinking_on_disk() -> None:
    """Apply enable_thinking=False inside hf_tokenizer/hf_processor for all Ray worker processes."""
    if not _rlsd_disable_thinking_enabled():
        return
    path = _find_module_file("verl.utils.tokenizer", os.path.join("verl", "utils", "tokenizer.py"))
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if _RLSD_DISABLE_THINKING_TOKENIZER_MARKER in text:
        return

    helper = '''
def _rlsd_wrap_disable_thinking(tokenizer):
    """Install Olmo chat template and force enable_thinking=False for Qwen3."""
    import os as _rlsd_os

    tokenizer = _rlsd_wrap_olmo_chat_template(tokenizer)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return tokenizer
    if getattr(tokenizer.apply_chat_template, "_rlsd_disable_thinking", False):
        return tokenizer
    if _rlsd_os.environ.get("DISABLE_THINKING_IN_CHAT_TEMPLATE", "").strip().lower() not in {
        "1", "true", "yes", "y", "on",
    }:
        return tokenizer

    try:
        from verl_rlsd.qwen3_chat_template import install_qwen3_no_think_chat_template
        install_qwen3_no_think_chat_template(tokenizer)
    except Exception:
        pass

    _orig_apply_chat = tokenizer.apply_chat_template

    def _apply_chat_no_think(messages, *args, **kwargs):
        kw = dict(kwargs)
        kw["enable_thinking"] = False
        try:
            out = _orig_apply_chat(messages, *args, **kw)
        except TypeError:
            kw.pop("enable_thinking", None)
            out = _orig_apply_chat(messages, *args, **kw)
        if (
            isinstance(out, str)
            and _rlsd_os.environ.get("STRIP_EMPTY_THINKING_GENERATION_PROMPT", "false").strip().lower()
            in {"1", "true", "yes", "on"}
            and kw.get("add_generation_prompt", True)
        ):
            try:
                from verl_rlsd.qwen3_chat_template import strip_empty_thinking_generation_prompt
                out = strip_empty_thinking_generation_prompt(out)
            except Exception:
                pass
        return out

    _apply_chat_no_think._rlsd_disable_thinking = True
    tokenizer.apply_chat_template = _apply_chat_no_think
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and inner is not tokenizer and hasattr(inner, "apply_chat_template"):
        _rlsd_wrap_disable_thinking(inner)
    return tokenizer

''' + _RLSD_DISABLE_THINKING_TOKENIZER_MARKER + "\n"

    anchor = "def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):"
    if anchor not in text:
        return
    text = text.replace(anchor, helper + anchor, 1)

    old_return = """    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor"""
    new_return = """    if correct_pad_token:
        set_pad_token_id(tokenizer)
    tokenizer = _rlsd_wrap_disable_thinking(tokenizer)
    return tokenizer


def hf_processor"""
    if old_return not in text:
        return
    text = text.replace(old_return, new_return, 1)

    old_proc_return = """    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor"""
    new_proc_return = """    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    processor = _rlsd_wrap_disable_thinking(processor)
    return processor"""
    if old_proc_return not in text:
        return
    text = text.replace(old_proc_return, new_proc_return, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _patch_hf_tokenizer_strip_empty_thinking_on_disk() -> None:
    path = _find_module_file("verl.utils.tokenizer", os.path.join("verl", "utils", "tokenizer.py"))
    if not path or _RLSD_STRIP_EMPTY_THINKING_PATCH in open(path, encoding="utf-8").read():
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if _APPLY_CHAT_NO_THINK_OLD not in text:
        return
    text = text.replace(_APPLY_CHAT_NO_THINK_OLD, _APPLY_CHAT_NO_THINK_NEW, 1)
    text = text.replace(
        _RLSD_DISABLE_THINKING_TOKENIZER_MARKER,
        _RLSD_STRIP_EMPTY_THINKING_PATCH + "\n" + _RLSD_DISABLE_THINKING_TOKENIZER_MARKER,
        1,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _patch_hf_tokenizer_install_no_think_template_on_disk() -> None:
    """Upgrade Ray-worker tokenizer patch to install Qwen3 no-think chat template."""
    if not _rlsd_disable_thinking_enabled():
        return
    path = _find_module_file("verl.utils.tokenizer", os.path.join("verl", "utils", "tokenizer.py"))
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if _RLSD_INSTALL_NO_THINK_CHAT_TEMPLATE_PATCH in text:
        return
    anchor = """    if _rlsd_os.environ.get("DISABLE_THINKING_IN_CHAT_TEMPLATE", "").strip().lower() not in {
        "1", "true", "yes", "y", "on",
    }:
        return tokenizer

    _orig_apply_chat = tokenizer.apply_chat_template"""
    insert = """    if _rlsd_os.environ.get("DISABLE_THINKING_IN_CHAT_TEMPLATE", "").strip().lower() not in {
        "1", "true", "yes", "y", "on",
    }:
        return tokenizer

    try:
        from verl_rlsd.qwen3_chat_template import install_qwen3_no_think_chat_template
        install_qwen3_no_think_chat_template(tokenizer)
    except Exception:
        pass
""" + _RLSD_INSTALL_NO_THINK_CHAT_TEMPLATE_PATCH + """

    _orig_apply_chat = tokenizer.apply_chat_template"""
    if anchor not in text:
        return
    text = text.replace(anchor, insert, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _repair_vllm_olmo_chat_template_patch() -> None:
    """Ensure async vLLM server injects Olmo chat template (Base has none on disk)."""
    path = _find_module_file(
        "verl.workers.rollout.vllm_rollout.vllm_async_server",
        os.path.join("verl", "workers", "rollout", "vllm_rollout", "vllm_async_server.py"),
    )
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if "load_olmo_chat_template" in text:
        return

    old = (
        "        _rlsd_chat_template = None\n"
        "        if os.environ.get(\"STRIP_EMPTY_THINKING_GENERATION_PROMPT\", \"true\").strip().lower() in {\n"
        "            \"1\", \"true\", \"yes\", \"on\",\n"
        "        }:\n"
        "            try:\n"
        "                from verl_rlsd.qwen3_chat_template import load_qwen3_chat_template_without_empty_thinking\n"
        "                _rlsd_chat_template = load_qwen3_chat_template_without_empty_thinking(local_path)\n"
        "            except Exception as _rlsd_ct_exc:\n"
        "                print(f\"[chat_template] failed to load stripped Qwen3 template: {_rlsd_ct_exc}\", flush=True)\n"
    )
    alt_old = old.replace('"true"', '"false"')
    new = (
        "        _rlsd_chat_template = None\n"
        "        if \"olmo\" in str(local_path).lower():\n"
        "            try:\n"
        "                from verl_rlsd.olmo_chat_template import load_olmo_chat_template\n"
        "                _rlsd_chat_template = load_olmo_chat_template(local_path)\n"
        "                print(f\"[chat_template] using Olmo chat template for {local_path}\", flush=True)\n"
        "            except Exception as _rlsd_ct_exc:\n"
        "                print(f\"[chat_template] failed to load Olmo template: {_rlsd_ct_exc}\", flush=True)\n"
        "        elif os.environ.get(\"STRIP_EMPTY_THINKING_GENERATION_PROMPT\", \"false\").strip().lower() in {\n"
        "            \"1\", \"true\", \"yes\", \"on\",\n"
        "        }:\n"
        "            try:\n"
        "                from verl_rlsd.qwen3_chat_template import load_qwen3_chat_template_without_empty_thinking\n"
        "                _rlsd_chat_template = load_qwen3_chat_template_without_empty_thinking(local_path)\n"
        "            except Exception as _rlsd_ct_exc:\n"
        "                print(f\"[chat_template] failed to load stripped Qwen3 template: {_rlsd_ct_exc}\", flush=True)\n"
    )
    if old in text:
        text = text.replace(old, new, 1)
    elif alt_old in text:
        text = text.replace(alt_old, new, 1)
    else:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _patch_vllm_openai_chat_template_on_disk() -> None:
    path = _find_module_file(
        "verl.workers.rollout.vllm_rollout.vllm_async_server",
        os.path.join("verl", "workers", "rollout", "vllm_rollout", "vllm_async_server.py"),
    )
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if _RLSD_VLLM_CHAT_TEMPLATE_PATCH not in text:
        anchor = "        # build serving chat\n        model_config = self.engine.model_config"
        if anchor not in text:
            return
        insert = (
            "        # build serving chat\n"
            "        _rlsd_chat_template = None\n"
            "        if \"olmo\" in str(local_path).lower():\n"
            "            try:\n"
            "                from verl_rlsd.olmo_chat_template import load_olmo_chat_template\n"
            "                _rlsd_chat_template = load_olmo_chat_template(local_path)\n"
            "                print(f\"[chat_template] using Olmo chat template for {local_path}\", flush=True)\n"
            "            except Exception as _rlsd_ct_exc:\n"
            "                print(f\"[chat_template] failed to load Olmo template: {_rlsd_ct_exc}\", flush=True)\n"
            "        elif os.environ.get(\"STRIP_EMPTY_THINKING_GENERATION_PROMPT\", \"false\").strip().lower() in {\n"
            "            \"1\", \"true\", \"yes\", \"on\",\n"
            "        }:\n"
            "            try:\n"
            "                from verl_rlsd.qwen3_chat_template import load_qwen3_chat_template_without_empty_thinking\n"
            "                _rlsd_chat_template = load_qwen3_chat_template_without_empty_thinking(local_path)\n"
            "            except Exception as _rlsd_ct_exc:\n"
            "                print(f\"[chat_template] failed to load stripped Qwen3 template: {_rlsd_ct_exc}\", flush=True)\n"
            "        model_config = self.engine.model_config"
            + _RLSD_VLLM_CHAT_TEMPLATE_PATCH
            + "\n"
        )
        text = text.replace(anchor, insert, 1)
        text = text.replace("            chat_template=None,", "            chat_template=_rlsd_chat_template,", 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    _repair_vllm_olmo_chat_template_patch()


_CAST_WORKER_BOOTSTRAP_MARKER = "# CAST_WORKER_BOOTSTRAP"
_CAST_WORKER_BOOTSTRAP_SNIPPET = """

# CAST_WORKER_BOOTSTRAP
# Must run at end-of-module so ActorRolloutRefWorker already exists.
try:
    from verl_rlsd.bootstrap import ensure_cast_worker_patches

    ensure_cast_worker_patches()
except Exception:
    pass
"""


def _patch_cast_worker_bootstrap_on_disk() -> None:
    """Ensure Ray GPU worker processes apply CAST internal-teacher patches after class defs."""
    targets = (
        ("verl.workers.fsdp_workers", os.path.join("verl", "workers", "fsdp_workers.py")),
        ("verl.workers.megatron_workers", os.path.join("verl", "workers", "megatron_workers.py")),
        ("verl.workers.engine_workers", os.path.join("verl", "workers", "engine_workers.py")),
    )
    for module_name, relative_path in targets:
        path = _find_module_file(module_name, relative_path)
        if not path or not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if _CAST_WORKER_BOOTSTRAP_MARKER in text:
            # Migrate old mid-file bootstrap (too early; classes not defined yet).
            if "Must run at end-of-module" not in text:
                old = (
                    "\n# CAST_WORKER_BOOTSTRAP\n"
                    "try:\n"
                    "    from verl_rlsd.bootstrap import ensure_cast_worker_patches\n"
                    "\n"
                    "    ensure_cast_worker_patches()\n"
                    "except Exception:\n"
                    "    pass\n"
                )
                text = text.replace(old, "\n", 1)
                if not text.endswith("\n"):
                    text += "\n"
                text += _CAST_WORKER_BOOTSTRAP_SNIPPET.lstrip("\n")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"[CAST] moved worker bootstrap to end of {path}", flush=True)
            continue
        if not text.endswith("\n"):
            text += "\n"
        text += _CAST_WORKER_BOOTSTRAP_SNIPPET.lstrip("\n")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[CAST] installed worker bootstrap at EOF in {path}", flush=True)


_CAST_TASKRUNNER_BOOTSTRAP_MARKER = "# CAST_TASKRUNNER_BOOTSTRAP"
_CAST_TASKRUNNER_BOOTSTRAP_SNIPPET = """
        # CAST_TASKRUNNER_BOOTSTRAP
        try:
            from verl_rlsd.bootstrap import ensure_cast_patches

            ensure_cast_patches()
        except Exception:
            pass
"""


def _patch_cast_taskrunner_bootstrap_on_disk() -> None:
    """Ensure Ray TaskRunner applies CAST advantage routing before trainer init."""
    path = _find_module_file(
        "verl.trainer.main_ppo",
        os.path.join("verl", "trainer", "main_ppo.py"),
    )
    if not path or not os.path.isfile(path):
        print("[CAST] skip TaskRunner bootstrap patch: main_ppo.py not found", flush=True)
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if _CAST_TASKRUNNER_BOOTSTRAP_MARKER in text:
        return
    anchor = "    def run(self, config):\n        # Print the initial configuration."
    if anchor not in text:
        print(f"[CAST] skip TaskRunner bootstrap patch for {path}: run() anchor missing", flush=True)
        return
    text = text.replace(
        anchor,
        "    def run(self, config):" + _CAST_TASKRUNNER_BOOTSTRAP_SNIPPET + "\n        # Print the initial configuration.",
        1,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[CAST] installed TaskRunner bootstrap in {path}", flush=True)


def _patch_ray_init_for_vllm() -> None:
    """Ensure Ray workers inherit the vLLM mode and training cluster RAY_ADDRESS."""
    os.environ.setdefault("VLLM_USE_V1", "1")
    try:
        import ray
    except Exception:
        return
    if getattr(ray.init, "_rlsd_vllm_env_patched", False):
        return
    original_init = ray.init

    def patched_init(*args, **kwargs):
        runtime_env = dict(kwargs.get("runtime_env") or {})
        env_vars = dict(runtime_env.get("env_vars") or {})
        env_vars.setdefault("VLLM_USE_V1", os.environ.get("VLLM_USE_V1", "1"))
        env_vars.setdefault(
            "TORCH_CUDA_ARCH_LIST",
            os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0"),
        )
        if os.environ.get("RLSD_MERGE_LORA_FOR_ASYNC_VLLM"):
            env_vars.setdefault(
                "RLSD_MERGE_LORA_FOR_ASYNC_VLLM",
                os.environ["RLSD_MERGE_LORA_FOR_ASYNC_VLLM"],
            )
        if os.environ.get("PYTHONPATH"):
            env_vars.setdefault("PYTHONPATH", os.environ["PYTHONPATH"])
        if os.environ.get("RAY_ADDRESS"):
            env_vars.setdefault("RAY_ADDRESS", os.environ["RAY_ADDRESS"])
        if os.environ.get("RAY_TMPDIR"):
            env_vars.setdefault("RAY_TMPDIR", os.environ["RAY_TMPDIR"])
        if os.environ.get("TMPDIR"):
            env_vars.setdefault("TMPDIR", os.environ["TMPDIR"])
        if os.environ.get("CAST_REAL_HOME"):
            env_vars.setdefault("CAST_REAL_HOME", os.environ["CAST_REAL_HOME"])
        if os.environ.get("HOME"):
            env_vars.setdefault("HOME", os.environ["HOME"])
        if os.environ.get("DISABLE_THINKING_IN_CHAT_TEMPLATE"):
            env_vars.setdefault(
                "DISABLE_THINKING_IN_CHAT_TEMPLATE",
                os.environ["DISABLE_THINKING_IN_CHAT_TEMPLATE"],
            )
        if os.environ.get("STRIP_EMPTY_THINKING_GENERATION_PROMPT"):
            env_vars.setdefault(
                "STRIP_EMPTY_THINKING_GENERATION_PROMPT",
                os.environ["STRIP_EMPTY_THINKING_GENERATION_PROMPT"],
            )
        if os.environ.get("MATH_PROMPT_PREFIX"):
            env_vars.setdefault("MATH_PROMPT_PREFIX", os.environ["MATH_PROMPT_PREFIX"])
        if os.environ.get("MATH_PROMPT_SUFFIX"):
            env_vars.setdefault("MATH_PROMPT_SUFFIX", os.environ["MATH_PROMPT_SUFFIX"])
        if os.environ.get("STRIP_DAPO_PROMPT_BOILERPLATE"):
            env_vars.setdefault(
                "STRIP_DAPO_PROMPT_BOILERPLATE",
                os.environ["STRIP_DAPO_PROMPT_BOILERPLATE"],
            )
        for _env_key in (
            "RLSD_INTERNAL_TEACHER_LOGPROB",
            "RLSD_INTERNAL_TEACHER_STRICT",
            "RLSD_INTERNAL_TEACHER_SNAPSHOT_MODE",
            "RLSD_INTERNAL_TEACHER_BASE_UNTIL_STEP",
            "RLSD_INTERNAL_TEACHER_SNAPSHOT_INTERVAL",
            "RLSD_TEACHER_PROMPT_MODE",
            "RLSD_MAX_TEACHER_PROMPT_LENGTH",
            "RLSD_TEACHER_LOGPROB_RESPONSE_LENGTH_CAP",
            "TOKENIZER_PATH",
            "MODEL_PATH",
            "CAST_WORKER_ENV_FILE",
            # wandb resume: TaskRunner calls wandb.init(); these must reach Ray actors.
            "WANDB_MODE",
            "WANDB_DIR",
            "WANDB_PROJECT",
            "WANDB_ENTITY",
            "WANDB_NAME",
            "WANDB_RUN_GROUP",
            "WANDB_RUN_ID",
            "WANDB_RESUME",
            "WANDB_API_KEY",
            "WANDB_DATA_DIR",
            "WANDB_CACHE_DIR",
            "WANDB_ARTIFACT_DIR",
        ):
            if os.environ.get(_env_key):
                # Force overwrite so reconnecting to an existing Ray cluster still
                # picks up the latest CAST teacher settings.
                env_vars[_env_key] = os.environ[_env_key]
        if os.environ.get("CAST_VERIFY_JSON"):
            env_vars.setdefault("CAST_VERIFY_JSON", os.environ["CAST_VERIFY_JSON"])
        if os.environ.get("CAST_VERIFY_MAX_EXAMPLES"):
            env_vars.setdefault("CAST_VERIFY_MAX_EXAMPLES", os.environ["CAST_VERIFY_MAX_EXAMPLES"])
        runtime_env["env_vars"] = env_vars
        kwargs["runtime_env"] = runtime_env
        result = original_init(*args, **kwargs)
        try:
            gcs = ray.get_runtime_context().gcs_address
            if gcs:
                os.environ["RAY_ADDRESS"] = gcs
        except Exception:
            pass
        return result

    patched_init._rlsd_vllm_env_patched = True
    ray.init = patched_init


def _maybe_strip_empty_thinking_prompt(result: Any, kwargs: dict[str, Any]) -> Any:
    if not isinstance(result, str) or not strip_empty_thinking_enabled():
        return result
    add_generation_prompt = kwargs.get("add_generation_prompt")
    if add_generation_prompt is None:
        add_generation_prompt = True
    if add_generation_prompt:
        return strip_empty_thinking_generation_prompt(result)
    return result


def _disable_thinking_on_tokenizer(tokenizer) -> None:
    """Same monkey-patch as opsd_train_anchor_strict_split_flip_wrong_boost.py."""
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return
    maybe_install_olmo_chat_template(tokenizer)
    if getattr(tokenizer.apply_chat_template, "_rlsd_disable_thinking", False):
        return

    install_qwen3_no_think_chat_template(tokenizer)

    _orig_apply_chat = tokenizer.apply_chat_template

    def _apply_chat_no_think(messages, *args, **kwargs):
        kw = dict(kwargs)
        kw["enable_thinking"] = False
        try:
            out = _orig_apply_chat(messages, *args, **kw)
        except TypeError:
            kw.pop("enable_thinking", None)
            out = _orig_apply_chat(messages, *args, **kw)
        return _maybe_strip_empty_thinking_prompt(out, kw)

    _apply_chat_no_think._rlsd_disable_thinking = True
    tokenizer.apply_chat_template = _apply_chat_no_think


def _patch_disable_thinking_in_chat_template() -> None:
    if not _rlsd_disable_thinking_enabled():
        return
    try:
        tokenizer_mod = importlib.import_module("verl.utils.tokenizer")
    except Exception:
        return

    original_hf_tokenizer = tokenizer_mod.hf_tokenizer
    original_hf_processor = tokenizer_mod.hf_processor
    if getattr(original_hf_tokenizer, "_rlsd_disable_thinking", False):
        return

    def patched_hf_tokenizer(*args, **kwargs):
        tokenizer = original_hf_tokenizer(*args, **kwargs)
        _disable_thinking_on_tokenizer(tokenizer)
        inner = getattr(tokenizer, "tokenizer", None)
        if inner is not None and inner is not tokenizer:
            _disable_thinking_on_tokenizer(inner)
        return tokenizer

    def patched_hf_processor(*args, **kwargs):
        processor = original_hf_processor(*args, **kwargs)
        _disable_thinking_on_tokenizer(processor)
        return processor

    patched_hf_tokenizer._rlsd_disable_thinking = True
    patched_hf_processor._rlsd_disable_thinking = True
    tokenizer_mod.hf_tokenizer = patched_hf_tokenizer
    tokenizer_mod.hf_processor = patched_hf_processor
    try:
        utils_mod = importlib.import_module("verl.utils")
        utils_mod.hf_tokenizer = patched_hf_tokenizer
        utils_mod.hf_processor = patched_hf_processor
    except Exception:
        pass
    print("[chat_template] enable_thinking=False (disable_thinking_in_chat_template=True)", flush=True)


def main() -> None:
    _repair_corrupted_rlsd_tokenizer_patch()
    _repair_vllm_olmo_chat_template_patch()
    _patch_hf_tokenizer_olmo_chat_template_on_disk()
    _patch_hf_tokenizer_disable_thinking_on_disk()
    _patch_hf_tokenizer_strip_empty_thinking_on_disk()
    _patch_hf_tokenizer_install_no_think_template_on_disk()
    _patch_rl_dataset_apply_chat_template_kwargs_on_disk()
    _patch_chat_scheduler_disable_thinking_on_disk()
    _patch_chat_scheduler_quiet_rollout_logs_on_disk()
    _patch_disable_thinking_in_chat_template()
    _patch_verl_vllm_async_server_on_disk()
    _patch_vllm_openai_chat_template_on_disk()
    _patch_fsdp_vllm_add_lora_on_disk()
    _patch_ray_init_for_vllm()
    _patch_cast_worker_bootstrap_on_disk()
    _patch_cast_taskrunner_bootstrap_on_disk()
    _patch_compute_advantage()
    _patch_truncate_rollout_batch()
    _patch_truncate_before_logprob()
    patch_rollout_length_logging()
    patch_rollout_dump()
    patch_teacher_ema()
    module_name = os.environ.get("VERL_MAIN_MODULE", "verl.trainer.main_ppo")
    module = importlib.import_module(module_name)
    entry = getattr(module, "main", None)
    if entry is None:
        raise RuntimeError(f"{module_name} does not expose a main() function.")
    sys.exit(entry())


if __name__ == "__main__":
    main()
