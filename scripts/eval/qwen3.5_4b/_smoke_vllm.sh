#!/bin/bash
#SBATCH --job-name=smoke_q35_vllm
#SBATCH --output=log/eval/qwen3.5_4b/smoke/%x.%j.out
#SBATCH --partition=GPUA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
set -euo pipefail

MODEL=/gpfs/share/home/2501210611/labShare/2501210611/model/qwen35_4b
cd /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis
mkdir -p log/eval/qwen3.5_4b/smoke

set +u
source activate qwen3_5
set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Prefer CUDA 12.8 nvcc when present (FlashInfer JIT needs modern nvcc flags).
if [[ -d /usr/local/cuda-12.8 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
elif command -v module >/dev/null 2>&1; then
  module load cuda/12.8 2>/dev/null || true
fi
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import traceback
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams

model = "/gpfs/share/home/2501210611/labShare/2501210611/model/qwen35_4b"
print("[smoke] loading config...")
cfg = AutoConfig.from_pretrained(model)
print("[smoke] model_type=", cfg.model_type)

tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
msgs = [{"role": "user", "content": "What is 2+2? Reply with only the number."}]
prompt = tok.apply_chat_template(
    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
print("[smoke] prompt head:", repr(prompt[:200]))

kwargs = dict(
    model=model,
    tokenizer=model,
    trust_remote_code=True,
    tensor_parallel_size=1,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
    max_model_len=8192,
)
# Prefer text-only path if supported by this vLLM build.
for extra in ({"language_model_only": True}, {}):
    try:
        print("[smoke] trying LLM with", extra or "no extras")
        llm = LLM(**kwargs, **extra)
        break
    except TypeError as e:
        print("[smoke] TypeError:", e)
        llm = None
    except Exception as e:
        print("[smoke] LLM init failed:", type(e).__name__, e)
        traceback.print_exc()
        raise
else:
    raise SystemExit("LLM init failed for all kwargs variants")

sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=128, presence_penalty=1.5)
outs = llm.generate([prompt], sp)
print("[smoke] output:", outs[0].outputs[0].text[:1000])
print("[smoke] SUCCESS")
PY
