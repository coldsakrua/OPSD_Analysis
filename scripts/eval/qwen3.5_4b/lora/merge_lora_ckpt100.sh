#!/bin/bash
#SBATCH --job-name=merge_st_snt_tt_lora_q35_ckpt100
#SBATCH --output=log/eval/qwen3.5_4b/lora/merge/%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --exclude=gpua800n03,gpua800n10,gpua800n05,gpua800n07,gpua800n08
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
fi
cd "${BASE_DIR}"

set +u
source activate qwen3_5
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p log/eval/qwen3.5_4b/lora/merge

python << 'PY'
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

BASE = Path("/gpfs/share/home/2501210611/labShare/2501210611/model/qwen35_4b")
ASSETS = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
    "configuration.json",
)

pairs = [
    (
        Path("outputs/qwen3.5_4b/snt_tt_lora_clip005_lr5e6_openthoughts_q35_4b/3208262/checkpoint-100"),
        Path("outputs/qwen3.5_4b/snt_tt_lora_clip005_lr5e6_openthoughts_q35_4b/3208262/merged_ckpt100"),
    ),
    (
        Path("outputs/qwen3.5_4b/st_tt_lora_clip005_lr5e6_openthoughts_q35_4b/3208263/checkpoint-100"),
        Path("outputs/qwen3.5_4b/st_tt_lora_clip005_lr5e6_openthoughts_q35_4b/3208263/merged_ckpt100"),
    ),
]

tok = AutoTokenizer.from_pretrained(str(BASE), trust_remote_code=True)

for adapter, out in pairs:
    print(f"[merge] adapter={adapter} -> {out}", flush=True)
    if out.exists() and (out / "config.json").exists() and list(out.glob("*.safetensors")):
        print(f"[merge] skip existing {out}", flush=True)
        continue
    out.mkdir(parents=True, exist_ok=True)

    print("[merge] loading base (ImageTextToText / ConditionalGeneration)...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(BASE),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    print("[merge] loading adapter...", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter))
    missing = getattr(model, "_keys_to_ignore_on_save", None)
    # Sanity: adapter keys should match language_model path
    n_loaded = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
    print(f"[merge] lora params present={n_loaded}", flush=True)
    if n_loaded < 100:
        raise RuntimeError(f"Too few LoRA params loaded ({n_loaded}); key mismatch?")

    print("[merge] merge_and_unload...", flush=True)
    model = model.merge_and_unload()
    print("[merge] save_pretrained...", flush=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    for name in ASSETS:
        src, dst = BASE / name, out / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"[merge] copied {name}", flush=True)
    del model
    torch.cuda.empty_cache()
    print(f"[merge] done {out}", flush=True)
    print("files:", sorted(p.name for p in out.iterdir()), flush=True)

print("[merge] all done", flush=True)
PY
