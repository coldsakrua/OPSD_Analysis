#!/bin/bash
#SBATCH --job-name=pmi_olmo_merge
#SBATCH --output=log/eval/olmo_7b_think/lora/merge/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --exclude=gpua800n03,gpua800n10,gpua800n13,gpua800n21
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2:00:00
set -euo pipefail

# Merge Purified-OPSD LoRA adapter into Olmo-3-7B-Think dense weights for SGLang eval.

BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}
cd "${BASE_DIR}"
: "${CHECKPOINT_PATH:?}"
: "${MERGED_PATH:?}"
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/gpfs/share/home/2501210611/labShare/2501210611/model/olmo-3-7b-think}

mkdir -p "$(dirname "${MERGED_PATH}")" log/eval/olmo_7b_think/lora/merge

set +u
source activate sglang
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python - <<PY
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = Path("${BASE_MODEL_PATH}")
adapter = Path("${CHECKPOINT_PATH}")
out = Path("${MERGED_PATH}")

print(f"[merge] base={base}", flush=True)
print(f"[merge] adapter={adapter}", flush=True)
print(f"[merge] out={out}", flush=True)

if out.exists() and (out / "config.json").exists() and list(out.glob("*.safetensors")):
    print("[merge] already exists; skip", flush=True)
    raise SystemExit(0)

out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    str(base),
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda",
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(model, str(adapter))
model = model.merge_and_unload()
model.save_pretrained(str(out), safe_serialization=True)
tok.save_pretrained(str(out))
# Copy chat template / jinja if present on base.
for name in ("chat_template.jinja", "tokenizer.json", "special_tokens_map.json"):
    src = base / name
    dst = out / name
    if src.is_file() and not dst.exists():
        shutil.copy2(src, dst)
print("[merge] done", flush=True)
PY
