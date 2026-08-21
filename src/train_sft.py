#!/usr/bin/env python3
"""Full-parameter CE SFT for Qwen3 (think mode) — independent of OPSDTrainer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Local offline store at <repo>/wandb can shadow the installed wandb package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_WANDB = _REPO_ROOT / "wandb"
if _LOCAL_WANDB.is_dir() and not (_LOCAL_WANDB / "__init__.py").exists():
    _repo = str(_REPO_ROOT.resolve())
    sys.path[:] = [
        p
        for p in sys.path
        if p not in ("", ".") and str(Path(p or ".").resolve()) != _repo
    ]

from cuda_env import ensure_nvidia_cudart_on_ld_path

ensure_nvidia_cudart_on_ld_path()

import torch
from transformers import AutoModelForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

from sft_dataset import DEFAULT_CHAT_TEMPLATE_PATH, load_sft_dataset, load_sft_tokenizer


DEFAULT_MODEL = "/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-parameter CE SFT (think mode)")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument(
        "--chat-template-path",
        default=os.environ.get("CHAT_TEMPLATE_PATH", DEFAULT_CHAT_TEMPLATE_PATH),
        help="Chat template source (default: Qwen3-1.7B instruct, same as eval).",
    )
    parser.add_argument("--dataset-path", required=True, help="Preprocessed parquet path or glob")
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "outputs/sft"))
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", "sft_qwen3"))
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enable_thinking to Qwen3 chat template (default: True).",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="Max checkpoints to keep (default: None = keep all).",
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--deepspeed", default=None, help="Path to DeepSpeed JSON config")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--packing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sequence packing (default: False).",
    )
    parser.add_argument(
        "--report-to",
        default="wandb",
        help="Trainer report_to (default: wandb). Use 'none' to disable.",
    )
    parser.add_argument("--num-proc", type=int, default=None, help="Dataset map workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sft] model={args.model_path}")
    print(f"[sft] chat_template={args.chat_template_path}")
    print(f"[sft] dataset={args.dataset_path}")
    print(f"[sft] output={output_dir}")
    print(
        f"[sft] think={args.enable_thinking} max_length={args.max_length} "
        f"max_steps={args.max_steps} save_steps={args.save_steps}"
    )
    print(
        f"[sft] micro={args.per_device_batch_size} gas={args.gradient_accumulation_steps} "
        f"lr={args.learning_rate} packing={args.packing}"
    )

    tokenizer = load_sft_tokenizer(
        args.model_path,
        chat_template_path=args.chat_template_path,
    )

    train_dataset = load_sft_dataset(
        args.dataset_path,
        enable_thinking=args.enable_thinking,
        num_proc=args.num_proc,
    )
    print(f"[sft] train_examples={len(train_dataset)}")
    pretokenized = "input_ids" in train_dataset.column_names
    print(f"[sft] pretokenized={pretokenized} cols={train_dataset.column_names}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    report_to = [] if args.report_to in ("none", "None", "") else [args.report_to]

    # Pretokenized datasets already truncated; still set max_length for collator safety.
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=args.run_name,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_length=args.max_length,
        packing=args.packing,
        # prompt/completion format → loss only on completion (assistant turn).
        # Qwen3 templates lack {% generation %}, so assistant_only_loss is unsuitable.
        # Pretokenized data carries completion_mask for the same behavior.
        completion_only_loss=True,
        assistant_only_loss=False,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=report_to,
        seed=args.seed,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        optim="adamw_torch",
        dataset_kwargs={"skip_prepare_dataset": True} if pretokenized else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[sft] saved final model to {final_dir}")


if __name__ == "__main__":
    main()
