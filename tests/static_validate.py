from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


for path in [*ROOT.glob("src/*.py"), *ROOT.glob("eval/*.py")]:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

ds = json.loads((ROOT / "configs/deepspeed_zero3.json").read_text(encoding="utf-8"))
require(ds["zero_optimization"]["stage"] == 3, "DeepSpeed must use ZeRO-3")
require(ds["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"], "full saves required")

accelerate_yaml = (ROOT / "configs/accelerate_zero3.yaml").read_text(encoding="utf-8")
require("distributed_type: DEEPSPEED" in accelerate_yaml, "Accelerate must use DeepSpeed")
require("num_processes: 4" in accelerate_yaml, "Accelerate must launch four ranks")

train_source = (ROOT / "src/train_opsd.py").read_text(encoding="utf-8")
trainer_source = (ROOT / "src/opsd_trainer.py").read_text(encoding="utf-8")
collator_source = (ROOT / "src/data_collator.py").read_text(encoding="utf-8")
common = (ROOT / "scripts/train_common.sh").read_text(encoding="utf-8")
require('"attn_implementation": "sdpa"' in train_source, "student must use SDPA")
require('teacher_kwargs["attn_implementation"] = "sdpa"' in trainer_source, "teacher must use SDPA")
require("flash_attention" not in train_source.lower(), "training entry must not enable FlashAttention")
require("VLLM_ATTENTION_BACKEND=XFORMERS" in common, "vLLM must use XFormers")
require("--max-steps 100" in common and "--save-steps 25" in common, "step/save schedule mismatch")
require("--max-prompt-length 1024" in common, "prompt length mismatch")
require("--max-completion-length 8192" in common, "response length mismatch")
require('else "π"' in collator_source, "fixed wrong privileged answer must be π")
require("enable_thinking=self.student_thinking" in collator_source, "student template switch missing")
require("enable_thinking=self.teacher_thinking" in collator_source, "teacher template switch missing")
require("self.teacher_model.requires_grad_(False)" in trainer_source, "teacher must be frozen")
require("top_k_loss=None" in train_source, "full-vocabulary loss must not set top-k")

matrix = {
    "opsd_correct_nothink.sh": "correct 0",
    "opsd_correct_think.sh": "correct 1",
    "opsd_pi_nothink.sh": "pi 0",
    "opsd_pi_think.sh": "pi 1",
    "opsd_instruction_nothink.sh": "instruction 0",
    "opsd_instruction_think.sh": "instruction 1",
}
for name, expected in matrix.items():
    text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
    require("#SBATCH --gres=gpu:4" in text, f"{name}: must request four GPUs")
    require(expected in text, f"{name}: wrong experiment mode")

require((ROOT / "vendor/verl/verl_rlsd").is_dir(), "CAST verl.zip was not extracted")
require((ROOT / "vendor/OPSD_official/opsd_trainer.py").is_file(), "official OPSD reference missing")
print("static validation passed")
