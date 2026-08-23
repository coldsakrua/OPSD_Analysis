#!/usr/bin/env python3
"""Full-parameter CE SFT for Qwen3 (think mode) — independent of OPSDTrainer."""

from __future__ import annotations

import argparse
import json
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
from transformers import AutoModelForCausalLM, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from sft_dataset import DEFAULT_CHAT_TEMPLATE_PATH, load_sft_dataset, load_sft_tokenizer


DEFAULT_MODEL = "/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b-base"
WANDB_META_NAME = "wandb_run.json"


def resolve_resume_checkpoint(output_dir: Path, resume: str | None) -> str | None:
    """Resolve checkpoint path for Trainer.resume_from_checkpoint."""
    if not resume:
        return None

    resume = resume.strip()
    if resume in {"true", "True", "1", "latest"}:
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.rsplit("-", 1)[-1])
            if p.name.rsplit("-", 1)[-1].isdigit()
            else -1,
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint-* directories found under {output_dir}")
        return str(checkpoints[-1].resolve())

    checkpoint = Path(resume).expanduser()
    if not checkpoint.is_absolute():
        candidate = output_dir / checkpoint
        checkpoint = candidate if candidate.is_dir() else checkpoint.resolve()
    else:
        checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    if not (checkpoint / "trainer_state.json").exists():
        raise FileNotFoundError(
            f"Resume checkpoint missing trainer_state.json: {checkpoint}"
        )
    return str(checkpoint)


def log_resume_checkpoint(checkpoint: str) -> None:
    state_path = Path(checkpoint) / "trainer_state.json"
    with state_path.open(encoding="utf-8") as f:
        state = json.load(f)
    print(
        f"[sft] resume checkpoint={checkpoint} "
        f"global_step={state.get('global_step')} epoch={state.get('epoch')}"
    )


def wandb_meta_path(output_dir: Path) -> Path:
    return output_dir / WANDB_META_NAME


def load_wandb_meta(output_dir: Path) -> dict[str, str] | None:
    path = wandb_meta_path(output_dir)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    return {str(k): str(v) for k, v in data.items() if v is not None}


def save_wandb_meta(
    output_dir: Path,
    *,
    run_id: str,
    run_name: str | None = None,
    project: str | None = None,
    group: str | None = None,
) -> None:
    path = wandb_meta_path(output_dir)
    payload = {
        "id": run_id,
        "name": run_name,
        "project": project or os.environ.get("WANDB_PROJECT"),
        "group": group or os.environ.get("WANDB_RUN_GROUP"),
    }
    payload = {k: v for k, v in payload.items() if v}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[sft] saved wandb meta -> {path} id={run_id}")


def setup_wandb_resume(
    output_dir: Path,
    *,
    resume_training: bool,
    run_name: str,
    wandb_run_id: str | None,
) -> str:
    """Attach to the previous wandb curve when resuming training.

    Sets WANDB_RUN_ID / WANDB_RESUME so wandb.init continues the same offline/online run.
    Returns the run_name to pass into SFTConfig (prefer the original name).
    """
    meta = load_wandb_meta(output_dir) or {}
    run_id = (wandb_run_id or os.environ.get("WANDB_RUN_ID") or meta.get("id") or "").strip()
    if not resume_training:
        if run_id:
            # Allow pinning id for a fresh job (rare); still create a new curve unless resume env set.
            os.environ.setdefault("WANDB_RUN_ID", run_id)
        return run_name

    if not run_id:
        print(
            "[sft] warn: resume training but no WANDB_RUN_ID / wandb_run.json; "
            "wandb will start a new run"
        )
        return run_name

    os.environ["WANDB_RUN_ID"] = run_id
    os.environ.setdefault("WANDB_RESUME", "allow")
    if meta.get("name"):
        run_name = meta["name"]
    if meta.get("project"):
        os.environ.setdefault("WANDB_PROJECT", meta["project"])
    if meta.get("group"):
        os.environ.setdefault("WANDB_RUN_GROUP", meta["group"])
    print(
        f"[sft] wandb resume id={run_id} name={run_name} "
        f"resume={os.environ.get('WANDB_RESUME')} project={os.environ.get('WANDB_PROJECT')}"
    )
    return run_name


class PersistWandbRunCallback(TrainerCallback):
    """Persist wandb run id under output_dir so later resume can attach the same curve."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        try:
            import wandb

            run = wandb.run
            if run is None or not getattr(run, "id", None):
                return
            save_wandb_meta(
                self.output_dir,
                run_id=str(run.id),
                run_name=getattr(run, "name", None) or args.run_name,
                project=getattr(run, "project", None),
                group=os.environ.get("WANDB_RUN_GROUP"),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[sft] warn: failed to persist wandb meta: {exc}")


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
    parser.add_argument(
        "--resume-from-checkpoint",
        default=os.environ.get("RESUME_FROM_CHECKPOINT"),
        help=(
            "Resume training from a checkpoint directory. "
            "Use 'latest' to pick the newest checkpoint-* under output-dir."
        ),
    )
    parser.add_argument(
        "--wandb-run-id",
        default=os.environ.get("WANDB_RUN_ID"),
        help=(
            "Reuse this wandb run id when resuming so loss curves continue. "
            "If omitted, load from output-dir/wandb_run.json."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_from_checkpoint = resolve_resume_checkpoint(
        output_dir, args.resume_from_checkpoint
    )
    run_name = setup_wandb_resume(
        output_dir,
        resume_training=resume_from_checkpoint is not None,
        run_name=args.run_name,
        wandb_run_id=args.wandb_run_id,
    )

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
    print(f"[sft] run_name={run_name}")
    if resume_from_checkpoint:
        log_resume_checkpoint(resume_from_checkpoint)
    else:
        print("[sft] resume=disabled (fresh run)")

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
        run_name=run_name,
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
        callbacks=[PersistWandbRunCallback(output_dir)],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[sft] saved final model to {final_dir}")


if __name__ == "__main__":
    main()
