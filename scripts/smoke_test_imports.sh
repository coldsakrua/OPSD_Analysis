#!/bin/bash
set -euo pipefail

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}
cd "${BASE_DIR}"
source activate anchor
export PYTHONPATH="${BASE_DIR}/src:${BASE_DIR}/vendor/verl:${BASE_DIR}/eval:${PYTHONPATH:-}"
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS

python - <<'PY'
import importlib
from importlib.metadata import version

required = [
    "torch", "accelerate", "transformers", "trl", "datasets", "deepspeed",
    "vllm", "wandb", "xformers", "ray", "verl",
]
for name in required:
    module = importlib.import_module(name)
    module_version = getattr(module, "__version__", None)
    if module_version is None:
        try:
            module_version = version(name)
        except Exception:
            module_version = "unknown"
    print(f"[import-ok] {name} {module_version}")

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from trl.models import prepare_deepspeed
from trl.extras.vllm_client import VLLMClient
from opsd_config import OPSDConfig
from data_collator import SelfDistillationDataCollator
from opsd_trainer import OPSDTrainer
import train_opsd

print("[import-ok] local OPSD modules")
print("[backend] training attention is explicitly configured as sdpa")
print("[backend] vLLM attention backend is XFORMERS")
PY
