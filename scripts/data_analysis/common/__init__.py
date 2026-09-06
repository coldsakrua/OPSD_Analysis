"""Shared utilities for OPSD student/teacher analysis."""

from .model_registry import (
    COMBOS,
    LONG_TEACHER_MAX_PROMPT,
    MODELS,
    TEACHER_PREFIXES,
    dataset_path,
    get_model_config,
    teacher_prefix_dataset,
    teacher_prefix_max_prompt,
)
from .metrics import compute_jsd_kl, compute_snr, compute_topk_kl, log_ratio_k1

__all__ = [
    "COMBOS",
    "LONG_TEACHER_MAX_PROMPT",
    "MODELS",
    "TEACHER_PREFIXES",
    "compute_jsd_kl",
    "compute_snr",
    "compute_topk_kl",
    "dataset_path",
    "get_model_config",
    "log_ratio_k1",
    "teacher_prefix_dataset",
    "teacher_prefix_max_prompt",
]
