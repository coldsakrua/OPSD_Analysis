"""Dataset helpers for classic CE SFT on OpenMathReasoning-style parquets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Eval/chat-template reference (Qwen3-1.7B instruct), same as scripts/eval/1.7b.
DEFAULT_CHAT_TEMPLATE_PATH = (
    "/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-1.7b"
)

# Same instruction as OPSD / eval / GRPO (Qwen3 math).
MATH_BOXED_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def append_math_instruction(problem: str, *, instruction: str = MATH_BOXED_INSTRUCTION) -> str:
    """Append boxed instruction to user problem if not already present."""
    text = str(problem or "").strip()
    instr = (instruction or "").strip()
    if not instr:
        return text
    if instr in text:
        return text
    if not text:
        return instr
    return f"{text}\n\n{instr}"


def load_sft_tokenizer(
    model_path: str,
    *,
    chat_template_path: str | None = DEFAULT_CHAT_TEMPLATE_PATH,
) -> PreTrainedTokenizerBase:
    """Load tokenizer from model weights, overlaying the eval chat template.

    Weights stay on ``model_path`` (e.g. qwen3-1.7b-base); chat template follows
    Qwen3-1.7B instruct used in eval (ModelScope: Qwen/Qwen3-1.7B).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if chat_template_path and chat_template_path != model_path:
        ref = AutoTokenizer.from_pretrained(chat_template_path, trust_remote_code=True)
        if not ref.chat_template:
            raise ValueError(f"No chat_template in {chat_template_path}")
        tokenizer.chat_template = ref.chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def expand_dataset_path(path: str) -> str | list[str]:
    """Expand a glob into a sorted file list for datasets.load_dataset."""
    p = Path(path)
    if any(ch in path for ch in "*?[]"):
        parent = p.parent if str(p.parent) != "" else Path(".")
        matches = sorted(str(x.resolve()) for x in parent.glob(p.name) if x.is_file())
        if not matches:
            raise FileNotFoundError(f"No files matched dataset glob: {path}")
        return matches
    resolved = p.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset path not found: {resolved}")
    return str(resolved)


def _extract_problem_solution(row: dict[str, Any]) -> tuple[str, str]:
    if "messages" in row and row["messages"] is not None:
        messages = row["messages"]
        if isinstance(messages, list) and len(messages) >= 2:
            user_parts = [str(m["content"]) for m in messages if m.get("role") == "user"]
            asst_parts = [str(m["content"]) for m in messages if m.get("role") == "assistant"]
            if user_parts and asst_parts:
                return user_parts[-1], asst_parts[-1]

    problem = row.get("problem")
    solution = row.get("generated_solution")
    if solution is None:
        solution = row.get("solution")
    if not problem or not solution or str(solution).strip() in ("", "n/a"):
        raise ValueError(
            "Row needs `messages`, prompt/completion, or non-empty "
            "`problem` + `generated_solution`/`solution`."
        )
    return str(problem), str(solution)


def normalize_sft_row(
    row: dict[str, Any],
    *,
    enable_thinking: bool = True,
    add_math_instruction: bool = True,
) -> dict[str, Any]:
    """Normalize to TRL prompt-completion conversational format.

    Qwen3 chat templates lack `{% generation %}`, so we use prompt/completion
    + completion_only_loss instead of assistant_only_loss.

    By default appends the Qwen3 math boxed instruction to the user turn
    (same string as OPSD / eval / GRPO).
    """
    if (
        isinstance(row.get("prompt"), list)
        and isinstance(row.get("completion"), list)
        and row["prompt"]
        and row["completion"]
    ):
        prompt = [{"role": str(m["role"]), "content": str(m["content"])} for m in row["prompt"]]
        completion = [
            {"role": str(m["role"]), "content": str(m["content"])} for m in row["completion"]
        ]
        if add_math_instruction:
            # Only touch the last user message.
            for i in range(len(prompt) - 1, -1, -1):
                if str(prompt[i].get("role", "")).lower() == "user":
                    prompt[i] = {
                        **prompt[i],
                        "content": append_math_instruction(str(prompt[i].get("content", ""))),
                    }
                    break
    else:
        problem, solution = _extract_problem_solution(row)
        if add_math_instruction:
            problem = append_math_instruction(problem)
        prompt = [{"role": "user", "content": problem}]
        completion = [{"role": "assistant", "content": solution}]

    out: dict[str, Any] = {
        "prompt": prompt,
        "completion": completion,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    for key in (
        "problem",
        "generated_solution",
        "expected_answer",
        "problem_type",
        "pass_rate_72b_tir",
        "problem_source",
        "generation_model",
        "inference_mode",
    ):
        if key in row and row[key] is not None:
            out[key] = row[key]
    return out


def load_sft_dataset(
    dataset_path: str,
    *,
    enable_thinking: bool = True,
    num_proc: int | None = None,
) -> Dataset:
    """Load parquet / HF disk dataset for SFTTrainer.

    - If ``dataset_path`` is a ``save_to_disk`` dir with ``input_ids``, load as-is
      (offline tokenized; training skips re-tokenize).
    - Else load parquet, apply optional ``*.spread_perm.npy``, normalize to
      prompt/completion format.
    """
    path = Path(dataset_path).expanduser()
    # Pretokenized HF dataset directory
    if path.is_dir() and (
        (path / "dataset_info.json").is_file()
        or (path / "state.json").is_file()
        or any(path.glob("*.arrow"))
    ):
        from datasets import load_from_disk

        ds = load_from_disk(str(path))
        cols = set(ds.column_names)
        if "input_ids" not in cols:
            raise ValueError(f"Pretokenized dataset missing input_ids: {path} cols={ds.column_names}")
        print(f"[sft_dataset] loaded pretokenized {path} rows={len(ds)} cols={ds.column_names}")
        return ds

    data_files = expand_dataset_path(dataset_path)
    ds = load_dataset("parquet", data_files=data_files, split="train")

    # Optional spread order from sibling *.spread_perm.npy (built by build_sft_le12k_spread.py).
    if isinstance(data_files, str):
        perm_path = Path(data_files + ".spread_perm.npy")
        if not perm_path.is_file():
            perm_path = Path(data_files).with_suffix(Path(data_files).suffix + ".spread_perm.npy")
        if perm_path.is_file():
            import numpy as np

            perm = np.load(perm_path)
            if len(perm) != len(ds):
                raise ValueError(
                    f"spread_perm length {len(perm)} != dataset length {len(ds)} ({perm_path})"
                )
            print(f"[sft_dataset] applying spread_perm {perm_path} n={len(perm)}")
            ds = ds.select(perm.tolist())

    # Already tokenized parquet
    if "input_ids" in ds.column_names:
        print(f"[sft_dataset] parquet already has input_ids rows={len(ds)}")
        return ds

    nproc = num_proc if num_proc is not None else max(1, min(8, (os.cpu_count() or 4) // 2))

    def _map(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        n = len(next(iter(batch.values())))
        rows = [{k: batch[k][i] for k in batch} for i in range(n)]
        normalized = [normalize_sft_row(r, enable_thinking=enable_thinking) for r in rows]
        keys = set().union(*(r.keys() for r in normalized))
        return {k: [r.get(k) for r in normalized] for k in keys}

    return ds.map(_map, batched=True, batch_size=256, num_proc=nproc, desc="normalize_sft")
