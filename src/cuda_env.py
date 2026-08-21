"""CUDA runtime setup — must run before `import torch` (SGLang spawn re-imports train_opsd.py)."""

from __future__ import annotations

import ctypes
import os
import sys


def _pip_cudart_path() -> str | None:
    conda_prefix = os.environ.get("CONDA_PREFIX") or ""
    if not conda_prefix:
        return None
    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cudart = os.path.join(
        conda_prefix,
        "lib",
        py_tag,
        "site-packages",
        "nvidia",
        "cuda_runtime",
        "lib",
        "libcudart.so.12",
    )
    return cudart if os.path.isfile(cudart) else None


def _nvidia_lib_dirs() -> list[str]:
    conda_prefix = os.environ.get("CONDA_PREFIX") or ""
    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia_root = os.path.join(conda_prefix, "lib", py_tag, "site-packages", "nvidia")
    if not os.path.isdir(nvidia_root):
        return []
    extra: list[str] = []
    cuda_runtime_lib = os.path.join(nvidia_root, "cuda_runtime", "lib")
    if os.path.isdir(cuda_runtime_lib):
        extra.append(cuda_runtime_lib)
    try:
        for name in sorted(os.listdir(nvidia_root)):
            if name == "cuda_runtime":
                continue
            lib_dir = os.path.join(nvidia_root, name, "lib")
            if os.path.isdir(lib_dir):
                extra.append(lib_dir)
    except OSError:
        return extra
    return extra


def _jit_link_dirs() -> list[str]:
    """Dirs for `-lcudart` when SGLang JIT compiles kernels (uses -L$CONDA/lib64, often empty)."""
    conda_prefix = os.environ.get("CONDA_PREFIX") or ""
    dirs: list[str] = []
    cudart = _pip_cudart_path()
    if cudart:
        dirs.append(os.path.dirname(cudart))
    if conda_prefix:
        for sub in ("targets/x86_64-linux/lib", "lib"):
            path = os.path.join(conda_prefix, sub)
            if os.path.isdir(path):
                dirs.append(path)
    return dirs


def ensure_nvidia_cudart_on_ld_path() -> None:
    """Ensure pip nvidia/cuda_runtime libcudart wins at runtime; keep link dirs for JIT."""
    extra = _nvidia_lib_dirs()
    cudart = _pip_cudart_path()

    current = [p for p in (os.environ.get("LD_LIBRARY_PATH") or "").split(":") if p]
    extra_set = set(extra)
    # Prepend pip nvidia libs; keep conda lib for non-cudart deps (do not strip — breaks JIT -lcudart).
    rest = [p for p in current if p not in extra_set]
    if extra:
        os.environ["LD_LIBRARY_PATH"] = ":".join(extra + rest)

    link_dirs = _jit_link_dirs()
    if link_dirs:
        lib_path = [p for p in (os.environ.get("LIBRARY_PATH") or "").split(":") if p]
        lib_set = set(link_dirs)
        os.environ["LIBRARY_PATH"] = ":".join(link_dirs + [p for p in lib_path if p not in lib_set])

    if cudart and not getattr(ensure_nvidia_cudart_on_ld_path, "_cudart_preloaded", False):
        ctypes.CDLL(cudart, mode=ctypes.RTLD_GLOBAL)
        ensure_nvidia_cudart_on_ld_path._cudart_preloaded = True  # type: ignore[attr-defined]
