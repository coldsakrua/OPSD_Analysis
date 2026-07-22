"""vLLM general plugin: patch Ministral tokenizer loading in all worker processes."""

from __future__ import annotations


def init_plugin() -> None:
    from verl_rlsd.ministral_tokenizer import (
        patch_vllm_ministral_tokenizer,
        patch_vllm_ministral_weight_files,
        patch_vllm_ministral_weight_loading,
    )

    patch_vllm_ministral_tokenizer()
    patch_vllm_ministral_weight_files()
    patch_vllm_ministral_weight_loading()
