#!/bin/bash
#SBATCH --job-name=warmup_fi_samp
#SBATCH --output=log/train/qwen3.5_4b/warmup_flashinfer.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=0:30:00
#SBATCH --exclude=gpua800n13,gpua800n21,gpua800n22,gpua800n23,gpua800n02,gpua800n08,gpua800n09
#SBATCH --nodelist=gpua800n16
set -euo pipefail

# Prebuild shared flashinfer sampling JIT so training jobs don't race/rebuild
# with incompatible nvcc flags on heterogeneous A800 nodes.

source /gpfs/share/home/2501210611/miniconda3/etc/profile.d/conda.sh
conda activate qwen3_5

CACHE_DIR="${HOME}/.cache/flashinfer/0.6.7.post3/80/cached_ops/sampling"
echo "[warmup] host=$(hostname)"
nvcc --version || true
echo "[warmup] clearing sampling JIT dir for clean rebuild: ${CACHE_DIR}"
rm -rf "${CACHE_DIR}"

python - <<'PY'
import torch
from flashinfer.sampling import get_sampling_module, top_k_top_p_sampling_from_probs

print("[warmup] torch.cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0))
mod = get_sampling_module()
print("[warmup] sampling module loaded:", type(mod))

probs = torch.softmax(torch.randn(4, 256, device="cuda", dtype=torch.float32), dim=-1)
samples = top_k_top_p_sampling_from_probs(probs, top_k=20, top_p=0.95)
print("[warmup] sample ok:", samples.shape if hasattr(samples, "shape") else type(samples), samples)
print("[warmup] DONE")
PY

ls -la "${CACHE_DIR}" | head -20
test -x "${CACHE_DIR}/sampling.so"
if grep -q "static-global-template-stub" "${CACHE_DIR}/build.ninja"; then
  echo "[warmup] ERROR: build.ninja still contains unsupported nvcc flag" >&2
  exit 1
fi
echo "[warmup] build.ninja has no static-global-template-stub flag (good)"
echo "[warmup] sampling.so ready"
