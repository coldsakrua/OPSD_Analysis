from __future__ import annotations

import argparse
import os
import shutil
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

# SGLang scheduler subprocesses re-import this module (spawn); fix libcudart before torch.
ensure_nvidia_cudart_on_ld_path()

import torch
from transformers import AutoTokenizer, TrainerCallback, set_seed

# Enforce the no-FlashAttention requirement even when this entry point is
# invoked directly instead of through scripts/train/*.sh.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
os.environ.setdefault("VLLM_USE_V1", "0")

from data_collator import SelfDistillationDataCollator
from opsd_config import OPSDConfig
from opsd_dataset import load_training_dataset, normalize_dataset, prompt_length_filter_applied
from opsd_trainer import OPSDTrainer
from sft_dataset import load_sft_tokenizer


DEFAULT_MODEL = "/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b"

# Qwen3.5 multimodal assets that HF Trainer/tokenizer.save_pretrained omit.
_PROCESSOR_ASSET_NAMES = (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "merges.txt",
    "vocab.json",
)


def _copy_processor_assets(src_model: str | Path, dst_dir: str | Path) -> None:
    """Copy image/video processor + vocab files needed by SGLang AutoProcessor."""
    src = Path(src_model)
    dst = Path(dst_dir)
    if not src.is_dir() or not dst.is_dir():
        return
    for name in _PROCESSOR_ASSET_NAMES:
        src_f = src / name
        dst_f = dst / name
        if src_f.is_file() and not dst_f.exists():
            shutil.copy2(src_f, dst_f)


class _CopyProcessorAssetsCallback(TrainerCallback):
    """Backfill multimodal processor files into each Trainer checkpoint."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    def on_save(self, args, state, control, **kwargs):
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        _copy_processor_assets(self.model_path, ckpt)
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-parameter OPSD on Qwen3-4B")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument(
        "--chat-template-path",
        default=os.environ.get("CHAT_TEMPLATE_PATH"),
        help=(
            "Optional: load tokenizer vocab from --model-path but overlay chat_template "
            "from this checkpoint (e.g. qwen3-1.7b instruct over qwen3-1.7b-base weights)."
        ),
    )
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
            "correct_simple",
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
            "irrelevant_other_sol",
        ),
        required=True,
        help=(
            "Teacher template: correct/correct_simple/pi/instruction/opsd (with GT privilege), or "
            "same/encourage/irrelevant (no-GT), or "
            "same_trans/encourage_trans/irrelevant_trans (no-GT + transition, no reference solution), or "
            "sample_irrelevant_trans (per-row irrelevant_prefix + transition), or "
            "irrelevant_other_sol (unrelated problem_B+solution_B then problem_A; no 'B solves A' claim). "
            "correct_simple = problem + answer + Qwen3 boxed instruction."
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
        "--beta",
        type=float,
        default=0.0,
        help=(
            "Generalized JSD interpolation in [0, 1]. "
            "beta=0 → forward KL KL(teacher‖student); "
            "beta=1 → reverse KL KL(student‖teacher); "
            "beta=0.5 → symmetric JSD. Default 0.0 (forward KL)."
        ),
    )
    parser.add_argument(
        "--top-k-loss",
        type=int,
        default=None,
        help=(
            "If set, restrict JSD/KL to teacher top-K tokens (renormalized). "
            "Default None = full-vocabulary forward KL. Mutually exclusive with "
            "--use-thinking-machines-loss."
        ),
    )
    parser.add_argument(
        "--high-entropy-ratio",
        type=float,
        default=None,
        help=(
            "If set to a value in (0, 1), only the top-ρ fraction of highest-entropy "
            "response tokens per sequence contribute to loss (student entropy). "
            "Default None = all valid response tokens (existing behavior). "
            "Mutually exclusive with --low-entropy-ratio / --uniform-loss-tokens / "
            "--last-loss-tokens."
        ),
    )
    parser.add_argument(
        "--low-entropy-ratio",
        type=float,
        default=None,
        help=(
            "If set to a value in (0, 1), only the bottom-ρ fraction of lowest-entropy "
            "response tokens per sequence contribute to loss (student entropy). "
            "E.g. 0.2 = le20 (bottom-20% only), 0.8 = le80 (bottom-80%). "
            "Default None = all valid response tokens (existing behavior). "
            "Mutually exclusive with --high-entropy-ratio / --uniform-loss-tokens / "
            "--last-loss-tokens."
        ),
    )
    parser.add_argument(
        "--uniform-loss-tokens",
        type=int,
        default=None,
        help=(
            "If set, uniformly sample this many valid response tokens per sequence "
            "for loss (e.g. 256 with max_completion=1024 → uni256). "
            "Sequences shorter than K keep all valid tokens. "
            "Mutually exclusive with entropy / last-token selection."
        ),
    )
    parser.add_argument(
        "--last-loss-tokens",
        type=int,
        default=None,
        help=(
            "If set, only the last K valid response tokens per sequence contribute "
            "to loss (e.g. 256 with max_completion=1024 → last256). "
            "Sequences shorter than K keep all valid tokens. "
            "Mutually exclusive with entropy / uniform selection."
        ),
    )
    parser.add_argument(
        "--use-thinking-machines-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace full-vocab JSD/KL with sampled-token logprob-diff PG loss: "
            "loss = -E[(log π_T - log π_S).detach() * log π_S]. "
            "Mutually exclusive with --top-k-loss."
        ),
    )
    parser.add_argument(
        "--teacher-update-steps",
        type=int,
        default=None,
        help=(
            "If set (e.g. 25), hard-copy student weights into the frozen teacher every N "
            "optimizer steps. Default None keeps a fixed initial teacher."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.1,
        help="Default rollout / JSD temperature when student/teacher temps are unset (1.1).",
    )
    parser.add_argument(
        "--student-temperature",
        type=float,
        default=None,
        help="Student rollout + JSD softmax temperature (defaults to --temperature).",
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        default=None,
        help="Teacher JSD softmax temperature (defaults to --temperature).",
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
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
    parser.add_argument(
        "--sglang-sampling-backend",
        choices=("pytorch", "flashinfer"),
        default=os.environ.get("SGLANG_SAMPLING_BACKEND", "pytorch"),
        help="SGLang token sampling backend. pytorch avoids flashinfer JIT on incomplete CUDA/CCCL nodes.",
    )
    parser.add_argument(
        "--sglang-reasoning-parser",
        default=os.environ.get("SGLANG_REASONING_PARSER", ""),
        help="SGLang reasoning parser (e.g. deepseek-r1 for Falcon-H1R). Empty disables.",
    )
    parser.add_argument(
        "--sglang-disable-piecewise-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH", "0") in {"1", "true", "True", "yes"},
        help="Disable SGLang piecewise CUDA graph (required for Falcon-H1 Mamba layers).",
    )
    parser.add_argument("--deepspeed", default="configs/deepspeed_zero3.json")
    parser.add_argument(
        "--use-peft",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable LoRA/PEFT student updates (siyan-zhao/OPSD main setup). "
            "With --fixed-teacher, teacher forward uses disable_adapter() on the base model."
        ),
    )
    parser.add_argument(
        "--fixed-teacher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fixed teacher at init policy. Full-param: frozen teacher copy. "
            "PEFT: disable LoRA adapters (official OPSD). Default True."
        ),
    )
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank (official run_opsd_1b.sh: 64).")
    parser.add_argument(
        "--lora-alpha", type=int, default=128, help="LoRA alpha (official run_opsd_1b.sh: 128)."
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules (official Qwen3 list).",
    )
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
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError(f"--beta must be in [0, 1], got {args.beta}")
    if args.high_entropy_ratio is not None and args.high_entropy_ratio >= 1.0:
        args.high_entropy_ratio = None
    if args.high_entropy_ratio is not None and args.high_entropy_ratio <= 0:
        parser.error("--high-entropy-ratio must be in (0, 1) when set")
    if args.low_entropy_ratio is not None and args.low_entropy_ratio >= 1.0:
        args.low_entropy_ratio = None
    if args.low_entropy_ratio is not None and args.low_entropy_ratio <= 0:
        parser.error("--low-entropy-ratio must be in (0, 1) when set")
    if args.uniform_loss_tokens is not None and args.uniform_loss_tokens <= 0:
        parser.error("--uniform-loss-tokens must be a positive integer when set")
    if args.last_loss_tokens is not None and args.last_loss_tokens <= 0:
        parser.error("--last-loss-tokens must be a positive integer when set")
    token_select_count = sum(
        x is not None
        for x in (
            args.high_entropy_ratio,
            args.low_entropy_ratio,
            args.uniform_loss_tokens,
            args.last_loss_tokens,
        )
    )
    if token_select_count > 1:
        parser.error(
            "--high-entropy-ratio, --low-entropy-ratio, --uniform-loss-tokens, "
            "and --last-loss-tokens are mutually exclusive"
        )
    if args.top_k_loss is not None and args.top_k_loss <= 0:
        parser.error("--top-k-loss must be a positive integer when set")
    if args.use_thinking_machines_loss and args.top_k_loss is not None:
        parser.error("--use-thinking-machines-loss and --top-k-loss are mutually exclusive")
    if args.teacher_update_steps is not None and args.teacher_update_steps <= 0:
        parser.error("--teacher-update-steps must be a positive integer when set")
    if args.use_peft and args.fixed_teacher is False:
        # Allowed but unstable per official README; keep as explicit opt-in only.
        pass
    if args.fixed_teacher and not args.use_peft and args.teacher_update_steps:
        # full-param path already supports periodic hard sync; no conflict.
        pass
    if args.use_peft and args.teacher_update_steps is not None:
        parser.error("--teacher-update-steps is incompatible with --use-peft (use --fixed-teacher)")
    if args.use_peft and args.teacher_model_path != args.model_path:
        parser.error(
            "--use-peft fixed-teacher uses disable_adapter() on the student base; "
            "cross-model --teacher-model-path is not supported"
        )
    if args.use_peft and args.lora_r <= 0:
        parser.error("--lora-r must be positive when --use-peft is set")
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

    if args.chat_template_path:
        tokenizer = load_sft_tokenizer(
            args.model_path,
            chat_template_path=args.chat_template_path,
        )
        tokenizer.padding_side = "right"
    else:
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
        temperature=args.temperature,
        student_temperature=args.student_temperature,
        teacher_temperature=args.teacher_temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        beta=args.beta,
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
        sglang_sampling_backend=args.sglang_sampling_backend,
        sglang_context_length=max_length,
        sglang_enable_memory_saver=True,
        sglang_reasoning_parser=args.sglang_reasoning_parser or None,
        sglang_disable_piecewise_cuda_graph=bool(args.sglang_disable_piecewise_cuda_graph),
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

    peft_config = None
    if args.use_peft:
        from peft import LoraConfig

        target_modules = [m.strip() for m in str(args.lora_target_modules).split(",") if m.strip()]
        if not target_modules:
            raise ValueError("--lora-target-modules must list at least one module")
        peft_config = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    world = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = args.per_device_batch_size * args.gradient_accumulation_steps * world
    print(
        f"[config] mode={args.privilege_mode} privilege_field={args.teacher_privilege_field} "
        f"student={args.model_path} teacher={args.teacher_model_path} "
        f"student_thinking={args.student_thinking} teacher_thinking={args.teacher_thinking} "
        f"rollout_backend={args.rollout_backend} "
        f"use_peft={args.use_peft} fixed_teacher={args.fixed_teacher} "
        f"lora_r={args.lora_r if args.use_peft else None} "
        f"lora_alpha={args.lora_alpha if args.use_peft else None} "
        f"lr={args.learning_rate} beta={args.beta} jsd_token_clip={args.jsd_token_clip} "
        f"high_entropy_ratio={args.high_entropy_ratio} "
        f"low_entropy_ratio={args.low_entropy_ratio} "
        f"uniform_loss_tokens={args.uniform_loss_tokens} "
        f"last_loss_tokens={args.last_loss_tokens} "
        f"top_k_loss={args.top_k_loss} use_thinking_machines_loss={args.use_thinking_machines_loss} "
        f"teacher_update_steps={args.teacher_update_steps} "
        f"temp={args.temperature} "
        f"temp_s={args.student_temperature if args.student_temperature is not None else args.temperature} "
        f"temp_t={args.teacher_temperature if args.teacher_temperature is not None else args.temperature} "
        f"top_p={args.top_p} top_k={args.top_k} "
        f"min_p={args.min_p} presence_penalty={args.presence_penalty} "
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
        peft_config=peft_config,
        fixed_teacher=bool(args.fixed_teacher),
        use_thinking_machines_loss=args.use_thinking_machines_loss,
        top_k_loss=args.top_k_loss,
        jsd_token_clip=args.jsd_token_clip,
        high_entropy_ratio=args.high_entropy_ratio,
        low_entropy_ratio=args.low_entropy_ratio,
        uniform_loss_tokens=args.uniform_loss_tokens,
        last_loss_tokens=args.last_loss_tokens,
        teacher_update_steps=args.teacher_update_steps,
        student_thinking=args.student_thinking,
        teacher_thinking=args.teacher_thinking,
        teacher_model_path=args.teacher_model_path,
        callbacks=[_CopyProcessorAssetsCallback(args.model_path)],
    )
    trainer.train()
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    _copy_processor_assets(args.model_path, final_dir)


if __name__ == "__main__":
    main()
