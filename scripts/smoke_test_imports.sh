#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR=${BASE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${BASE_DIR}"
# conda activate scripts reference unset vars; keep nounset elsewhere
set +u
source activate anchor
set -u
export PYTHONPATH="${BASE_DIR}/src:${BASE_DIR}/vendor/verl:${BASE_DIR}/eval:${PYTHONPATH:-}"
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS

python - <<'PY'
import importlib
from importlib.metadata import version

required = [
    "torch", "accelerate", "transformers", "trl", "datasets", "deepspeed",
    "vllm", "wandb", "xformers", "ray", "verl", "pyarrow", "tqdm",
]
optional = ["math_verify"]
for name in required:
    module = importlib.import_module(name)
    module_version = getattr(module, "__version__", None)
    if module_version is None:
        try:
            module_version = version(name)
        except Exception:
            module_version = "unknown"
    print(f"[import-ok] {name} {module_version}")

for name in optional:
    try:
        module = importlib.import_module(name)
        module_version = getattr(module, "__version__", None) or "unknown"
        print(f"[import-ok] {name} {module_version}")
    except ImportError as exc:
        print(f"[import-warn] {name} missing: {exc}")

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from trl.models import prepare_deepspeed
from trl.extras.vllm_client import VLLMClient
from opsd_config import OPSDConfig
from data_collator import SelfDistillationDataCollator
from opsd_trainer import OPSDTrainer
import train_opsd
import eval_math_vllm_local
import data_utils
import reward_fn

print("[import-ok] local OPSD train modules")
print("[import-ok] local OPSD eval modules")
print("[backend] training attention is explicitly configured as sdpa")
print("[backend] vLLM attention backend is XFORMERS")
PY
