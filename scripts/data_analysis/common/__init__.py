"""Shared utilities for OPSD student/teacher analysis."""

from .model_registry import COMBOS, MODELS, dataset_path, get_model_config, teacher_prefix_dataset
from .metrics import compute_jsd_kl, compute_topk_kl, log_ratio_k1

__all__ = [
    "COMBOS",
    "MODELS",
    "compute_jsd_kl",
    "compute_topk_kl",
    "dataset_path",
    "get_model_config",
    "log_ratio_k1",
    "teacher_prefix_dataset",
]
