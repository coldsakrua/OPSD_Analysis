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
require("num_processes: 2" in accelerate_yaml, "Accelerate must launch two ranks")

train_source = (ROOT / "src/train_opsd.py").read_text(encoding="utf-8")
trainer_source = (ROOT / "src/opsd_trainer.py").read_text(encoding="utf-8")
collator_source = (ROOT / "src/data_collator.py").read_text(encoding="utf-8")
require('"attn_implementation": "sdpa"' in train_source, "student must use SDPA")
require('teacher_kwargs["attn_implementation"] = "sdpa"' in trainer_source, "teacher must use SDPA")
require("flash_attention" not in train_source.lower(), "training entry must not enable FlashAttention")
require('else "π"' in collator_source, "fixed wrong privileged answer must be π")
require("enable_thinking=self.student_thinking" in collator_source, "student template switch missing")
require("enable_thinking=self.teacher_thinking" in collator_source, "teacher template switch missing")
require("self.teacher_model.requires_grad_(False)" in trainer_source, "teacher must be frozen")
require("left_align_prompt_completion" in trainer_source, "trainer must left-align prompt/completion packing")
require((ROOT / "src/opsd_sequence.py").is_file(), "opsd_sequence helper missing")
require((ROOT / "tests/test_token_alignment.py").is_file(), "token alignment tests missing")
official_hyper = (
    ROOT / "scripts/train/qwen3_1.7b/hyper/opsd_snt_tt_official_clip005_lr5e6.sh"
)
require(official_hyper.is_file(), "official-hyper ablation script missing")
official_hyper_text = official_hyper.read_text(encoding="utf-8")
require("JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}" in official_hyper_text, "official-hyper clip default wrong")
require("LEARNING_RATE=${LEARNING_RATE:-5e-6}" in official_hyper_text, "official-hyper lr default wrong")
require("--no-student-thinking" in official_hyper_text and "--teacher-thinking" in official_hyper_text, "official-hyper thinking flags wrong")
require('"--top-k-loss"' in train_source, "CLI must expose --top-k-loss")
require('"--use-thinking-machines-loss"' in train_source, "CLI must expose --use-thinking-machines-loss")
require("top_k_loss=args.top_k_loss" in train_source, "trainer must receive args.top_k_loss")
require(
    "use_thinking_machines_loss=args.use_thinking_machines_loss" in train_source,
    "trainer must receive args.use_thinking_machines_loss",
)

matrix = {
    "opsd_nothink_4b.sh": ("correct", "0"),
    "opsd_think_4b.sh": ("correct", "1"),
    "opsd_pi_nothink_4b.sh": ("pi", "0"),
    "opsd_pi_think_4b.sh": ("pi", "1"),
    "opsd_instruction_nothink_4b.sh": ("instruction", "0"),
    "opsd_instruction_think_4b.sh": ("instruction", "1"),
}
for name, (mode, thinking) in matrix.items():
    text = (ROOT / "scripts" / "train" / name).read_text(encoding="utf-8")
    require("#SBATCH --gres=gpu:2" in text, f"{name}: must request two GPUs")
    require(f"MODE={mode}" in text, f"{name}: wrong privilege mode")
    require(f"THINKING={thinking}" in text, f"{name}: wrong thinking flag")
    require("train_common.sh" not in text, f"{name}: must be self-contained")
    require("VLLM_ATTENTION_BACKEND=XFORMERS" in text, f"{name}: vLLM must use XFormers")
    require("--max-steps 100" in text and "--save-steps 25" in text, f"{name}: step/save schedule mismatch")
    require("--max-prompt-length 1024" in text, f"{name}: prompt length mismatch")
    require("--max-completion-length 1024" in text, f"{name}: response length mismatch")
    require("--privilege-mode" in text, f"{name}: privilege-mode arg missing")
    require("accelerate launch" in text, f"{name}: must launch training directly")

asymmetric = (ROOT / "scripts/train/opsd_student_nothink_teacher_think_4b.sh").read_text(encoding="utf-8")
require("#SBATCH --gres=gpu:2" in asymmetric, "asymmetric script must request two GPUs")
require("MODE=correct" in asymmetric, "asymmetric script must use correct privilege")
require("STUDENT_THINKING=0" in asymmetric and "TEACHER_THINKING=1" in asymmetric, "asymmetric thinking flags wrong")
require("--no-student-thinking" in asymmetric and "--teacher-thinking" in asymmetric, "asymmetric CLI flags missing")
require("VLLM_ATTENTION_BACKEND=XFORMERS" in asymmetric, "asymmetric script must use XFormers")
require("--student-thinking" in train_source and "--teacher-thinking" in train_source, "train entry must expose split thinking flags")

for name, thinking in (("eval_nothink.sh", "0"), ("eval_think.sh", "1")):
    text = (ROOT / "scripts" / "eval" / name).read_text(encoding="utf-8")
    require("#SBATCH --gres=gpu:1" in text, f"{name}: must request one GPU")
    require(f"THINKING={thinking}" in text, f"{name}: wrong thinking flag")
    require("eval_common.sh" not in text, f"{name}: must be self-contained")
    require("eval_math_vllm_local.py" in text, f"{name}: must call eval entrypoint")

require(not (ROOT / "scripts/train_common.sh").exists(), "train_common.sh should be removed")
require(not (ROOT / "scripts/eval_common.sh").exists(), "eval_common.sh should be removed")
require((ROOT / "vendor/verl/verl_rlsd").is_dir(), "CAST verl.zip was not extracted")
opsd_official = ROOT / "vendor/OPSD_official/opsd_trainer.py"
trl_ref = ROOT / "vendor/trl_v0.22.1"
if not opsd_official.is_file():
    print("warning: vendor/OPSD_official reference snapshot is empty/missing")
if not any(trl_ref.glob("*")):
    print("warning: vendor/trl_v0.22.1 reference snapshot is empty/missing")
print("static validation passed")
