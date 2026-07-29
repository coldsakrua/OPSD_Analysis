from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, set_seed

# Enforce the no-FlashAttention requirement even when this entry point is
# invoked directly instead of through scripts/train/*.sh.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
os.environ.setdefault("VLLM_USE_V1", "0")

from data_collator import SelfDistillationDataCollator
from opsd_config import OPSDConfig
from opsd_dataset import load_training_dataset, normalize_dataset, prompt_length_filter_applied
from opsd_trainer import OPSDTrainer


DEFAULT_MODEL = "/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-parameter OPSD on Qwen3-4B")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument(
        "--teacher-model-path",
        default=os.environ.get("TEACHER_MODEL_PATH"),
        help=(
            "Optional frozen teacher checkpoint. Defaults to --model-path (self-distillation). "
            "Use a different path for cross-model distillation "
            "(e.g. student=Instruct, teacher=qwen3-4b think)."
        ),
    )
    parser.add_argument("--dataset-path", default=os.environ.get("DATASET_PATH"), required=False)
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "outputs/opsd"))
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", "opsd_qwen3_4b"))
    parser.add_argument(
        "--privilege-mode",
        choices=(
            "correct",
            "pi",
            "instruction",
            "opsd",
            "same",
            "encourage",
            "irrelevant",
            "same_trans",
            "encourage_trans",
            "irrelevant_trans",
            "sample_irrelevant_trans",
        ),
        required=True,
        help=(
            "Teacher template: correct/pi/instruction/opsd (with GT privilege), or "
            "same/encourage/irrelevant (no-GT), or "
            "same_trans/encourage_trans/irrelevant_trans (no-GT + transition, no reference solution), or "
            "sample_irrelevant_trans (per-row irrelevant_prefix + transition)."
        ),
    )
    parser.add_argument(
        "--teacher-privilege-field",
        choices=("solution", "answer", "none"),
        default="solution",
        help=(
            "Teacher privilege content: full trajectory (`solution`), short answer (`answer`), "
            "or `none` for no-GT modes."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Convenience flag: turn on thinking for both student and teacher.",
    )
    parser.add_argument(
        "--student-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Qwen3 enable_thinking for the student prompt (default: follows --enable-thinking).",
    )
    parser.add_argument(
        "--teacher-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Qwen3 enable_thinking for the teacher prompt (default: follows --enable-thinking).",
    )
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--jsd-token-clip", type=float, default=1e-6,
        help=(
            "Per-token JSD clip (siyan-zhao/OPSD default is 0.05). "
            "Set <=0 to disable clipping. Current repo default 1e-6 is very aggressive."
        ),
    )
    parser.add_argument(
        "--rollout-backend",
        choices=("vllm", "sglang"),
        default=os.environ.get("ROLLOUT_BACKEND", "vllm"),
        help="On-policy generation backend. Use sglang for Olmo-3.",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument(
        "--sglang-mem-fraction-static",
        type=float,
        default=float(os.environ.get("SGLANG_MEM_FRACTION_STATIC", "0.40")),
    )
    parser.add_argument(
        "--sglang-attention-backend",
        default=os.environ.get("SGLANG_ATTENTION_BACKEND", "triton"),
    )
    parser.add_argument("--deepspeed", default="configs/deepspeed_zero3.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.dataset_path:
        parser.error("--dataset-path or DATASET_PATH is required")
    if args.student_thinking is None:
        args.student_thinking = bool(args.enable_thinking)
    if args.teacher_thinking is None:
        args.teacher_thinking = bool(args.enable_thinking)
    if not args.teacher_model_path:
        args.teacher_model_path = args.model_path
    # <=0 disables clip (trainer expects None).
    if args.jsd_token_clip is not None and args.jsd_token_clip <= 0:
        args.jsd_token_clip = None
    return args


def _maybe_install_olmo_chat_template(tokenizer) -> None:
    try:
        from verl_rlsd.olmo_chat_template import maybe_install_olmo_chat_template
    except ImportError:
        return
    maybe_install_olmo_chat_template(tokenizer)


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
    _maybe_install_olmo_chat_template(tokenizer)

    teacher_tokenizer = tokenizer
    if args.teacher_model_path != args.model_path:
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            args.teacher_model_path,
            trust_remote_code=True,
            padding_side="right",
        )
        if teacher_tokenizer.pad_token_id is None:
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        _maybe_install_olmo_chat_template(teacher_tokenizer)

    collator = SelfDistillationDataCollator(
        tokenizer=tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        max_length=max_length,
        max_prompt_length=args.max_prompt_length,
        privilege_mode=args.privilege_mode,
        teacher_privilege_field=args.teacher_privilege_field,
        student_thinking=args.student_thinking,
        teacher_thinking=args.teacher_thinking,
    )
    train_dataset = normalize_dataset(load_training_dataset(args.dataset_path))
    before = len(train_dataset)
    # Offline length meta is only trusted for same-model self-distillation.
    offline_ok = (
        args.teacher_model_path == args.model_path
        and prompt_length_filter_applied(
            args.dataset_path,
            privilege_mode=args.privilege_mode,
            student_thinking=args.student_thinking,
            teacher_thinking=args.teacher_thinking,
            max_prompt_length=args.max_prompt_length,
            model_path=args.model_path,
            teacher_privilege_field=args.teacher_privilege_field,
        )
    )
    if offline_ok:
        print(
            f"[dataset] prompt length already filtered offline; keep {before} examples",
            flush=True,
        )
    else:
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
        rollout_backend=args.rollout_backend,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=1,
        vllm_sync_frequency=1,
        vllm_enable_sleep_mode=True,
        sglang_mem_fraction_static=args.sglang_mem_fraction_static,
        sglang_attention_backend=args.sglang_attention_backend,
        sglang_context_length=max_length,
        sglang_enable_memory_saver=True,
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

    world = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = args.per_device_batch_size * args.gradient_accumulation_steps * world
    print(
        f"[config] mode={args.privilege_mode} privilege_field={args.teacher_privilege_field} "
        f"student={args.model_path} teacher={args.teacher_model_path} "
        f"student_thinking={args.student_thinking} teacher_thinking={args.teacher_thinking} "
        f"rollout_backend={args.rollout_backend} "
        f"lr={args.learning_rate} jsd_token_clip={args.jsd_token_clip} "
        f"global_batch={global_batch} "
        f"(micro={args.per_device_batch_size} gas={args.gradient_accumulation_steps} world={world}) "
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
        jsd_token_clip=args.jsd_token_clip,
        student_thinking=args.student_thinking,
        teacher_thinking=args.teacher_thinking,
        teacher_model_path=args.teacher_model_path,
    )
    trainer.train()
    trainer.save_model(str(Path(args.output_dir) / "final"))
    tokenizer.save_pretrained(str(Path(args.output_dir) / "final"))


if __name__ == "__main__":
    main()
