#!/bin/bash
# Compute nodes default to GCC 8.5; PyTorch / sglang JIT (c_dlpack) needs GCC >= 9.
# Source this after `source activate sglang`.
if command -v module >/dev/null 2>&1; then
  module load gcc/11.4.0 2>/dev/null \
    || module load gcc/11 2>/dev/null \
    || module load gcc/9.5.0 2>/dev/null \
    || module load gcc/9 2>/dev/null \
    || true
fi
hash -r 2>/dev/null || true
export CC="${CC:-$(command -v gcc || true)}"
export CXX="${CXX:-$(command -v g++ || true)}"
echo "[sglang] gcc=$(${CXX:-g++} --version 2>/dev/null | head -1)"
