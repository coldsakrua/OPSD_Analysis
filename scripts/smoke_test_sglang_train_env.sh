#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR=${BASE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${BASE_DIR}"
set +u
source activate sglang
set -u
export PYTHONPATH="${BASE_DIR}/src:${BASE_DIR}/vendor/verl:${BASE_DIR}/eval:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python - <<'PY'
import importlib
from importlib.metadata import version

required = [
    "torch",
    "accelerate",
    "transformers",
    "trl",
    "datasets",
    "deepspeed",
    "ray",
    "sglang",
    "wandb",
    "peft",
    "pyarrow",
    "tqdm",
]
optional = ["verl", "math_verify"]

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
        module_version = getattr(module, "__version__", None)
        if module_version is None:
            try:
                module_version = version(name)
            except Exception:
                module_version = "unknown"
        print(f"[import-ok] {name} {module_version}")
    except ImportError as exc:
        print(f"[import-warn] {name} missing: {exc}")

from sglang import Engine  # noqa: F401
from trl.models import prepare_deepspeed  # noqa: F401
from verl_rlsd.olmo_chat_template import maybe_install_olmo_chat_template  # noqa: F401
from opsd_config import OPSDConfig  # noqa: F401
from data_collator import SelfDistillationDataCollator  # noqa: F401
from opsd_trainer import OPSDTrainer  # noqa: F401
import train_opsd  # noqa: F401

print("[import-ok] local OPSD train modules")
print("[import-ok] sglang.Engine + verl_rlsd.olmo_chat_template")
print("[backend] training attention: sdpa; rollout: sglang (triton)")
assert getattr(train_opsd, "parse_args", None) is not None
print("[smoke] sglang train env OK")
PY
