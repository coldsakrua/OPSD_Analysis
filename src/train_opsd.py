from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, set_seed

# Enforce the no-FlashAttention requirement even when this entry point is
# invoked directly instead of through scripts/train_common.sh.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
os.environ.setdefault("VLLM_USE_V1", "0")

from data_collator import SelfDistillationDataCollator
from opsd_config import OPSDConfig
from opsd_trainer import OPSDTrainer


DEFAULT_MODEL = "/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b"


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(x for x in parts if x).strip()
    return "" if content is None else str(content).strip()


def _extract_problem(prompt: Any) -> str:
    if isinstance(prompt, list):
        users = [x for x in prompt if isinstance(x, dict) and str(x.get("role", "user")).lower() == "user"]
        text = _content_to_text((users[-1] if users else prompt[-1]).get("content", "")) if prompt else ""
    elif isinstance(prompt, dict):
        text = _content_to_text(prompt.get("content", prompt))
    else:
        text = str(prompt or "").strip()
    text = re.sub(
        r"^\s*Solve\s+the\s+following(?:\s+math)?\s+problem\s*,?\s+step\s+by\s+step\s*[.:]?\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"\s*Please\s+reason\s+step\s+by\s+step.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return text.strip()


def _extract_solution(row: dict[str, Any]) -> str:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"]).strip()
    for key in ("ground_truth", "answer", "solution", "target"):
        if row.get(key) is not None and str(row[key]).strip():
            return str(row[key]).strip()
    return ""


def _load_training_dataset(path: str):
    source = Path(path)
    if source.is_dir() and (source / "dataset_info.json").exists():
        dataset = load_from_disk(str(source))
        return dataset["train"] if hasattr(dataset, "keys") and "train" in dataset else dataset
    files = [x.strip() for x in path.split(",") if x.strip()]
    return load_dataset("parquet", data_files=files, split="train")


def _normalize_dataset(dataset):
    def convert(row: dict[str, Any]) -> dict[str, str]:
        prompt = row.get("prompt", row.get("problem", row.get("question", "")))
        return {"problem": _extract_problem(prompt), "solution": _extract_solution(row)}

    dataset = dataset.map(convert, desc="Normalizing DAPO fields")
    return dataset.filter(
        lambda row: bool(str(row["problem"]).strip()) and bool(str(row["solution"]).strip()),
        desc="Dropping empty problems/answers",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-parameter OPSD on Qwen3-4B")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--dataset-path", default=os.environ.get("DATASET_PATH"), required=False)
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "outputs/opsd"))
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", "opsd_qwen3_4b"))
    parser.add_argument("--privilege-mode", choices=("correct", "pi", "instruction"), required=True)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=8192)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--deepspeed", default="configs/deepspeed_zero3.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.dataset_path:
        parser.error("--dataset-path or DATASET_PATH is required")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    max_length = args.max_prompt_length + args.max_completion_length

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = SelfDistillationDataCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        max_prompt_length=args.max_prompt_length,
        privilege_mode=args.privilege_mode,
        student_thinking=args.enable_thinking,
        teacher_thinking=args.enable_thinking,
    )
    train_dataset = _normalize_dataset(_load_training_dataset(args.dataset_path))
    before = len(train_dataset)
    train_dataset = train_dataset.filter(collator.fits, desc="Enforcing student/teacher prompt length")
    print(f"[dataset] prompt cap kept {len(train_dataset)}/{before} examples", flush=True)
    if len(train_dataset) == 0:
        raise RuntimeError("no training rows remain after prompt filtering")

    training_args = OPSDConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=5,
        logging_steps=1,
        logging_strategy="steps",
        eval_strategy="no",
        report_to=["wandb"],
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.0,
        max_grad_norm=0.1,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        tf32=True,
        optim="adamw_torch",
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=max_length,
        max_completion_length=args.max_completion_length,
        temperature=1.1,
        top_p=0.95,
        top_k=20,
        beta=0.0,
        lmbda=1.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=1,
        vllm_sync_frequency=1,
        vllm_enable_sleep_mode=True,
        steps_per_generation=args.gradient_accumulation_steps,
        log_completions=False,
        wandb_project=os.environ.get("WANDB_PROJECT", "OPSD"),
        wandb_run_group=os.environ.get("WANDB_RUN_GROUP"),
        seed=args.seed,
        data_seed=args.seed,
        model_init_kwargs={
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
            "torch_dtype": torch.bfloat16,
            "use_cache": False,
            "low_cpu_mem_usage": True,
        },
    )

    print(
        f"[config] mode={args.privilege_mode} thinking={args.enable_thinking} "
        f"global_batch={args.per_device_batch_size * args.gradient_accumulation_steps * 4} "
        f"prompt={args.max_prompt_length} response={args.max_completion_length}",
        flush=True,
    )
    trainer = OPSDTrainer(
        model=args.model_path,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=None,
        fixed_teacher=True,
        use_thinking_machines_loss=False,
        top_k_loss=None,
        jsd_token_clip=1e-6,
        student_thinking=args.enable_thinking,
        teacher_thinking=args.enable_thinking,
    )
    trainer.train()
    trainer.save_model(str(Path(args.output_dir) / "final"))
    tokenizer.save_pretrained(str(Path(args.output_dir) / "final"))


if __name__ == "__main__":
    main()
