from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from verl.utils.dataset.rl_dataset import RLHFDataset

from data_utils import DEFAULT_MATH_INSTRUCTION_SUFFIX, _strip_math_prompt_boilerplate

DEFAULT_MATH_PROMPT_PREFIX = ""
DEFAULT_MATH_PROMPT_SUFFIX = DEFAULT_MATH_INSTRUCTION_SUFFIX


def _strip_enabled() -> bool:
    return os.environ.get("STRIP_DAPO_PROMPT_BOILERPLATE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _math_prompt_prefix() -> str:
    return os.environ.get("MATH_PROMPT_PREFIX", DEFAULT_MATH_PROMPT_PREFIX).strip()


def _math_prompt_suffix() -> str:
    return os.environ.get("MATH_PROMPT_SUFFIX", DEFAULT_MATH_PROMPT_SUFFIX).strip()


def _has_boxed_instruction(text: str) -> bool:
    low = text.lower()
    return "final answer" in low and "boxed" in low


def _format_math_instruction_text(
    text: str,
    *,
    add_prefix: bool = True,
    add_suffix: bool = True,
) -> str:
    text = text.strip()
    prefix = _math_prompt_prefix() if add_prefix else ""
    if prefix and prefix.lower() not in text.lower():
        text = f"{prefix}\n\n{text}"
    suffix = _math_prompt_suffix() if add_suffix else ""
    if suffix and not _has_boxed_instruction(text):
        text = f"{text}\n\n{suffix}"
    return text


def _normalize_math_content(content: Any) -> Any:
    strip_enabled = _strip_enabled()
    if isinstance(content, str):
        if strip_enabled:
            content = _strip_math_prompt_boilerplate(content)
        return _format_math_instruction_text(content)
    if isinstance(content, list):
        out: list[Any] = []
        text_indices: list[int] = []
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "text":
                item = dict(part)
                text = str(item.get("text", ""))
                if strip_enabled:
                    text = _strip_math_prompt_boilerplate(text)
                item["text"] = text
                text_indices.append(len(out))
                out.append(item)
            else:
                out.append(part)
        if text_indices:
            first_idx = text_indices[0]
            last_idx = text_indices[-1]
            if first_idx == last_idx:
                item = dict(out[first_idx])
                item["text"] = _format_math_instruction_text(str(item.get("text", "")))
                out[first_idx] = item
            else:
                first_item = dict(out[first_idx])
                first_item["text"] = _format_math_instruction_text(
                    str(first_item.get("text", "")),
                    add_suffix=False,
                )
                out[first_idx] = first_item
                last_item = dict(out[last_idx])
                last_item["text"] = _format_math_instruction_text(
                    str(last_item.get("text", "")),
                    add_prefix=False,
                )
                out[last_idx] = last_item
        return out
    return content


def normalize_user_prompt_messages(messages: list[Any]) -> list[Any]:
    """Strip DAPO boilerplate and append the boxed-answer math instruction."""
    out: list[Any] = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            out.append(msg)
            continue
        role = str(msg.get("role", "")).lower()
        if role != "user":
            out.append(msg)
            continue
        normalized = copy.deepcopy(msg)
        normalized["content"] = _normalize_math_content(msg.get("content", ""))
        out.append(normalized)
    return out


def strip_dapo_prompt_boilerplate(messages: list[Any]) -> list[Any]:
    """Remove baked-in DAPO math instruction wrappers from user turns."""
    return normalize_user_prompt_messages(messages)


def extract_gsm8k_final_answer(answer_text: str) -> str:
    """Extract the final GSM8K answer (typically after '####')."""
    text = str(answer_text or "").strip()
    if not text:
        return ""
    match = re.search(r"####\s*(.+?)\s*$", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def convert_gsm8k_parquet_to_rlsd(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Convert raw GSM8K parquet (question/answer) to veRL RLSD row schema."""
    import datasets

    input_path = str(input_path)
    output_path = str(output_path)
    output_parent = Path(output_path).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    dataframe = datasets.load_dataset("parquet", data_files=input_path)["train"]
    original_len = len(dataframe)
    print(f"[gsm8k] loaded {original_len} rows from {input_path}", flush=True)

    def convert_row(example: dict[str, Any], idx: int) -> dict[str, Any]:
        question = str(example.get("question", "")).strip()
        answer_full = str(example.get("answer", "")).strip()
        ground_truth = extract_gsm8k_final_answer(answer_full)
        return {
            "data_source": "gsm8k",
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"ground_truth": ground_truth, "style": "rule/gsm8k"},
            "solution": answer_full,
            "extra_info": {"index": str(example.get("id", idx)), "split": "train"},
        }

    converted = dataframe.map(
        convert_row,
        with_indices=True,
        desc="Converting GSM8K rows to RLSD schema",
    )
    converted = converted.filter(
        lambda row: bool(str(row.get("prompt", [{}])[0].get("content", "")).strip())
        and bool(str(row.get("reward_model", {}).get("ground_truth", "")).strip()),
        desc="Filtering empty GSM8K rows",
    )
    converted_len = len(converted)
    converted.to_parquet(output_path)
    meta = {
        "source": input_path,
        "output": output_path,
        "original_len": original_len,
        "converted_len": converted_len,
        "format": "gsm8k",
    }
    meta_path = f"{output_path}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[gsm8k] wrote {converted_len} rows -> {output_path}", flush=True)
    return meta


def normalize_prompt_field(row: Mapping[str, Any], prompt_key: str = "prompt") -> dict[str, Any]:
    out = dict(row)
    prompt = out.get(prompt_key)
    if isinstance(prompt, list):
        out[prompt_key] = strip_dapo_prompt_boilerplate(prompt)
    return out


def prompt_token_length(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    prompt_key: str = "prompt",
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
) -> int:
    messages = copy.deepcopy(list(row[prompt_key]))
    return len(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            **dict(apply_chat_template_kwargs or {}),
        )
    )


def preprocess_rlsd_dataset(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    tokenizer: Any,
    max_prompt_length: int = 1024,
    prompt_key: str = "prompt",
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
    num_proc: int = 8,
    batch_size: int = 1000,
) -> dict[str, Any]:
    """Normalize prompts, filter overlong rows, and save parquet for fast training startup."""
    import datasets

    input_path = str(input_path)
    output_path = str(output_path)
    output_parent = Path(output_path).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    dataframe = datasets.load_dataset("parquet", data_files=input_path)["train"]
    original_len = len(dataframe)
    print(f"[preprocess] loaded {original_len} rows from {input_path}", flush=True)

    chat_kwargs = dict(apply_chat_template_kwargs or {})
    dataframe = dataframe.map(
        lambda row: normalize_prompt_field(row, prompt_key=prompt_key),
        num_proc=num_proc,
        desc="Normalizing prompts",
        batch_size=batch_size,
    )

    def keep_row(row: dict[str, Any]) -> bool:
        return (
            prompt_token_length(
                row,
                tokenizer=tokenizer,
                prompt_key=prompt_key,
                apply_chat_template_kwargs=chat_kwargs,
            )
            <= max_prompt_length
        )

    dataframe = dataframe.filter(
        keep_row,
        num_proc=num_proc,
        desc=f"Filtering prompts longer than {max_prompt_length} tokens",
        batch_size=batch_size,
    )
    filtered_len = len(dataframe)
    print(f"[preprocess] kept {filtered_len}/{original_len} rows", flush=True)

    dataframe.to_parquet(output_path)
    meta = {
        "source": input_path,
        "output": output_path,
        "original_len": original_len,
        "filtered_len": filtered_len,
        "max_prompt_length": max_prompt_length,
        "prompt_key": prompt_key,
        "apply_chat_template_kwargs": chat_kwargs,
        "strip_dapo_prompt_boilerplate": _strip_enabled(),
        "math_prompt_prefix": _math_prompt_prefix(),
        "math_prompt_suffix": _math_prompt_suffix(),
    }
    meta_path = f"{output_path}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[preprocess] wrote {output_path}", flush=True)
    print(f"[preprocess] wrote {meta_path}", flush=True)
    return meta


class RLSDRLHFDataset(RLHFDataset):
    """RLHFDataset that strips DAPO-style instruction boilerplate from user prompts."""

    def __init__(self, data_files, tokenizer, config, processor=None):
        self.prompt_preprocessed = bool(config.get("prompt_preprocessed", False))
        if self.prompt_preprocessed:
            config["filter_overlong_prompts"] = False
        super().__init__(data_files, tokenizer, config, processor)

    def _build_messages(self, example: dict):
        # Use explicit parent dispatch: super() breaks when this class is loaded
        # via load_extern_type() and datasets.filter runs with num_proc > 0.
        messages = RLHFDataset._build_messages(self, copy.deepcopy(example))
        if self.prompt_preprocessed:
            return messages
        return strip_dapo_prompt_boilerplate(messages)

    def _read_files_and_tokenize(self):
        import datasets

        dataframes = []
        for parquet_file in self.data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")
        if self.prompt_preprocessed:
            print("prompt_preprocessed=true: skipping online prompt normalization/filter", flush=True)
            return

        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer

            def doc2len(doc) -> int:
                messages = self._build_messages(doc)
                return len(
                    tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        **self.apply_chat_template_kwargs,
                    )
                )

            self.dataframe = self.dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )
            print(f"filter dataset len: {len(self.dataframe)}")
