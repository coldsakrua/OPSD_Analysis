from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerFast

_MINISTRAL_CONFIG_REGISTERED = False
_VLLM_TOKENIZER_PATCHED = False
_VLLM_WEIGHT_PATCHED = False


def ensure_ministral_config_registered() -> None:
    """Register ministral3 config alias for transformers < 5.0."""
    global _MINISTRAL_CONFIG_REGISTERED
    if _MINISTRAL_CONFIG_REGISTERED:
        return
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.ministral.configuration_ministral import MinistralConfig

    if "ministral3" not in CONFIG_MAPPING:
        CONFIG_MAPPING.register("ministral3", MinistralConfig)
    _MINISTRAL_CONFIG_REGISTERED = True


def is_ministral_model_path(model_path: str) -> bool:
    path_lower = str(model_path).lower()
    if "ministral" in path_lower:
        return True
    tokenizer_config = Path(model_path) / "tokenizer_config.json"
    if not tokenizer_config.is_file():
        return False
    data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    return data.get("tokenizer_class") == "TokenizersBackend"


def load_ministral_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    """Load Ministral tokenizer by bypassing the TokenizersBackend class issue."""
    model_dir = Path(model_path)
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(model_dir / "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
        unk_token="<unk>",
    )
    jinja_path = model_dir / "chat_template.jinja"
    if jinja_path.is_file():
        tok.chat_template = jinja_path.read_text(encoding="utf-8")
    return tok


def load_eval_tokenizer(model_path: str, **kwargs: Any):
    if is_ministral_model_path(model_path):
        return load_ministral_tokenizer(model_path)
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, **kwargs)


def fix_ministral_hf_config(hf_config: Any) -> Any:
    """Ensure vLLM can resolve the Ministral text backbone architecture."""
    text_cfg = getattr(hf_config, "text_config", None)
    if text_cfg is None:
        return hf_config
    if text_cfg.architectures is None and getattr(text_cfg, "model_type", "") in (
        "ministral3",
        "mistral",
    ):
        text_cfg.architectures = ["MistralForCausalLM"]
    return hf_config


def patch_vllm_ministral_tokenizer() -> None:
    """Route vLLM tokenizer init through load_ministral_tokenizer for Ministral models."""
    global _VLLM_TOKENIZER_PATCHED
    if _VLLM_TOKENIZER_PATCHED:
        return
    from functools import lru_cache

    import vllm.transformers_utils.tokenizer as vllm_tokenizer_mod
    import vllm.transformers_utils.tokenizer_group as vllm_tokenizer_group_mod

    _orig_get_tokenizer = vllm_tokenizer_mod.get_tokenizer

    def _get_tokenizer(tokenizer_name, *args, **kwargs):
        if is_ministral_model_path(str(tokenizer_name)):
            tokenizer = load_ministral_tokenizer(str(tokenizer_name))
            return vllm_tokenizer_mod.get_cached_tokenizer(tokenizer)
        return _orig_get_tokenizer(tokenizer_name, *args, **kwargs)

    vllm_tokenizer_mod.get_tokenizer = _get_tokenizer
    vllm_tokenizer_mod.cached_get_tokenizer = lru_cache(_get_tokenizer)
    vllm_tokenizer_group_mod.get_tokenizer = _get_tokenizer

    from transformers import AutoTokenizer

    if not getattr(AutoTokenizer.from_pretrained, "_ministral_patched", False):
        _orig_from_pretrained = AutoTokenizer.from_pretrained.__func__

        @classmethod
        def _patched_from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
            if is_ministral_model_path(str(pretrained_model_name_or_path)):
                tokenizer = load_ministral_tokenizer(str(pretrained_model_name_or_path))
                return vllm_tokenizer_mod.get_cached_tokenizer(tokenizer)
            return _orig_from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)

        _patched_from_pretrained._ministral_patched = True  # type: ignore[attr-defined]
        AutoTokenizer.from_pretrained = _patched_from_pretrained

    _VLLM_TOKENIZER_PATCHED = True


def _remap_ministral_fp8_weight_name(name: str) -> str:
    if name.endswith(".activation_scale"):
        return name.replace(".activation_scale", ".input_scale")
    if name.endswith(".weight_scale_inv"):
        return name.replace(".weight_scale_inv", ".weight_scale")
    return name


def patch_vllm_ministral_weight_loading() -> None:
    """Remap HF FP8 Ministral scale tensor names to vLLM parameter names."""
    global _VLLM_WEIGHT_PATCHED
    if _VLLM_WEIGHT_PATCHED:
        return
    from vllm.model_executor.models import llama

    if getattr(llama.LlamaModel.load_weights, "_ministral_patched", False):
        _VLLM_WEIGHT_PATCHED = True
        return

    _orig_load_weights = llama.LlamaModel.load_weights

    def _patched_load_weights(self, weights):
        def _iter_weights():
            for name, weight in weights:
                yield _remap_ministral_fp8_weight_name(name), weight

        return _orig_load_weights(self, _iter_weights())

    _patched_load_weights._ministral_patched = True  # type: ignore[attr-defined]
    llama.LlamaModel.load_weights = _patched_load_weights
    _VLLM_WEIGHT_PATCHED = True


def patch_vllm_ministral_weight_files() -> None:
    """Load only HF-format model.safetensors, skip consolidated.safetensors."""
    from vllm.model_executor.model_loader.loader import DefaultModelLoader

    if getattr(DefaultModelLoader._prepare_weights, "_ministral_patched", False):
        return

    _orig_prepare_weights = DefaultModelLoader._prepare_weights

    def _patched_prepare_weights(
        self,
        model_name_or_path,
        revision,
        fall_back_to_pt,
        allow_patterns_overrides,
    ):
        hf_folder, hf_weights_files, use_safetensors = _orig_prepare_weights(
            self,
            model_name_or_path,
            revision,
            fall_back_to_pt,
            allow_patterns_overrides,
        )
        if not is_ministral_model_path(str(model_name_or_path)):
            return hf_folder, hf_weights_files, use_safetensors

        preferred = os.path.join(hf_folder, "model.safetensors")
        filtered = [
            path
            for path in hf_weights_files
            if "consolidated" not in os.path.basename(path)
        ]
        if os.path.isfile(preferred):
            hf_weights_files = [preferred]
        elif filtered:
            hf_weights_files = filtered
        return hf_folder, hf_weights_files, use_safetensors

    _patched_prepare_weights._ministral_patched = True  # type: ignore[attr-defined]
    DefaultModelLoader._prepare_weights = _patched_prepare_weights
