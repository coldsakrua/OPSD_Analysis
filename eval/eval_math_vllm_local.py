#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from verl_rlsd.ministral_tokenizer import (
    ensure_ministral_config_registered,
    fix_ministral_hf_config,
    is_ministral_model_path,
    load_eval_tokenizer,
    patch_vllm_ministral_tokenizer,
)
from verl_rlsd.olmo_chat_template import is_olmo_model_path, maybe_install_olmo_chat_template

ensure_ministral_config_registered()

try:
    from math_verify import parse, verify

    _HAS_MATH_VERIFY = True
except ImportError:
    _HAS_MATH_VERIFY = False


def extract_user_prompt(prompt_obj: Any) -> str:
    """
    Extract user prompt text from common chat/prompt container formats.
    """
    if prompt_obj is None:
        return ""
    if isinstance(prompt_obj, str):
        return prompt_obj.strip()
    if isinstance(prompt_obj, dict):
        content = prompt_obj.get("content")
        if isinstance(content, str):
            return content.strip()
        role = str(prompt_obj.get("role", "")).lower()
        if role == "user":
            text = prompt_obj.get("text")
            if isinstance(text, str):
                return text.strip()
        # Fallback: try nested message list.
        for key in ("messages", "prompt", "conversation"):
            nested = prompt_obj.get(key)
            text = extract_user_prompt(nested)
            if text:
                return text
        return ""
    if isinstance(prompt_obj, list):
        # Prefer explicit user turn.
        for item in prompt_obj:
            if isinstance(item, dict) and str(item.get("role", "")).lower() == "user":
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        # Fallback: first non-empty text-like field.
        for item in prompt_obj:
            text = extract_user_prompt(item)
            if text:
                return text
        return ""
    return str(prompt_obj).strip()


def max_seq_len_from_model_config(model_path: str) -> Optional[int]:
    """Return max context from config.json (vLLM refuses max_model_len above this unless env override)."""
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        m = getattr(cfg, "max_position_embeddings", None)
        if m is None:
            m = getattr(cfg, "model_max_length", None)
        if m is None:
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                m = getattr(text_cfg, "max_position_embeddings", None)
        return int(m) if m is not None else None
    except Exception:
        return None


def _is_gemma3_model(model_path: str) -> bool:
    p = str(model_path).lower()
    if "gemma-3" in p or "gemma3" in p:
        return True
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        mt = getattr(cfg, "model_type", "")
        if mt in ("gemma3", "gemma3_text"):
            return True
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and getattr(text_cfg, "model_type", "") == "gemma3_text":
            return True
    except Exception:
        pass
    return False


def _math_user_suffix(eval_type: str, gemma3: bool) -> str:
    if eval_type == "mcq":
        if gemma3:
            return (
                "\n\nProvide the final answer as a single capital letter "
                "(A, B, C, ...), wrapped in \\boxed{}."
            )
        return (
            "\n\nPlease reason step by step and provide the final answer as a single capital letter "
            "(A, B, C, ...), wrapped in \\boxed{}."
        )
    if gemma3:
        return "\n\nSolve the problem and put your final answer within \\boxed{}."
    return "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def _apply_chat_prompt(tokenizer: Any, messages: List[Dict[str, str]], enable_thinking: bool) -> str:
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        kwargs["enable_thinking"] = bool(enable_thinking)
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def extract_boxed_answer(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    num_left_braces = 0
    right_brace_idx = None
    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[7:-1].strip()
    return None


_RELAXED_MATH_ANSWER_PATTERNS = (
    r"[Tt]he answer is:?\s*\$([^$]+)\$",
    r"[Tt]he answer is:?\s*\\boxed\{([^}]+)\}",
    r"[Tt]he answer is:?\s*(\\frac\{[^}]+\}\{[^}]+\})",
    r"[Tt]he answer is:?\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)",
    r"[Ff]inal answer:?\s*\$([^$]+)\$",
    r"[Ff]inal answer:?\s*\\boxed\{([^}]+)\}",
    r"答案是:?\s*\$([^$]+)\$",
    r"答案是:?\s*\\boxed\{([^}]+)\}",
    r"答案是:?\s*(\d+)",
)


def extract_relaxed_math_answer(text: str) -> Optional[str]:
    """Prefer \\boxed{}; fall back to DeepSeek-style 'The answer is: ...' phrases."""
    if not text:
        return None
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    best: Optional[str] = None
    best_pos = -1
    for pat in _RELAXED_MATH_ANSWER_PATTERNS:
        for m in re.finditer(pat, text):
            if m.start() >= best_pos:
                best_pos = m.start()
                best = m.group(1).strip()
    return best


def extract_math_answer(text: str, *, relaxed: bool = False) -> Optional[str]:
    if relaxed:
        return extract_relaxed_math_answer(text)
    return extract_boxed_answer(text)


def normalize_math_ground_truth(ground_truth: str) -> str:
    """Match OLMo-Eval AIME/HMMT gold handling: unwrap \\boxed{} and strip leading zeros."""
    gt = str(ground_truth or "").strip()
    if not gt:
        return gt
    inner = extract_boxed_answer(gt)
    if inner is not None:
        gt = inner
    elif gt.startswith("\\boxed{") and gt.endswith("}"):
        gt = gt[7:-1].strip()
    if re.fullmatch(r"0*\d+", gt):
        gt = gt.lstrip("0") or "0"
    return gt


def grade_answer(predicted: Optional[str], ground_truth: str) -> bool:
    if predicted is None:
        return False
    gt = normalize_math_ground_truth(ground_truth)
    if _HAS_MATH_VERIFY:
        try:
            pred_w = predicted if "$" in predicted else f"${predicted}$"
            gt_w = gt if "$" in gt else f"${gt}$"
            pred_parsed = parse(pred_w, fallback_mode="no_fallback")
            gt_parsed = parse(gt_w, fallback_mode="no_fallback")
            return bool(verify(gt_parsed, pred_parsed, timeout_seconds=5))
        except Exception:
            pass
    pred_norm = predicted.replace("$", "").replace(" ", "").lower().strip()
    gt_norm = gt.replace("$", "").replace(" ", "").lower().strip()
    return pred_norm == gt_norm


def load_jsonl_examples(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            rows.append(
                {
                    "id": o.get("id", len(rows)),
                    "problem": str(o["problem"]).strip(),
                    "ground_truth": str(o["answer"]).strip(),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _extract_gsm8k_final_answer(answer_text: str) -> str:
    """
    Extract the final GSM8K answer.
    Typical format ends with: '#### 72'
    """
    text = str(answer_text or "").strip()
    if not text:
        return ""
    m = re.search(r"####\s*(.+?)\s*$", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: use last non-empty line if no #### marker exists.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def load_gsm8k_hf_examples(
    path: Path,
    limit: Optional[int],
    gsm8k_config: str = "main",
    gsm8k_split: str = "test",
) -> List[Dict[str, Any]]:
    """
    Load GSM8K from a HuggingFace-datasets style directory.
    Expected fields: question, answer
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Loading GSM8K directory requires `datasets`. Install with: pip install datasets"
        ) from e

    ds = load_dataset(str(path), gsm8k_config, split=gsm8k_split)
    rows: List[Dict[str, Any]] = []
    for i, o in enumerate(ds):
        problem = str(o.get("question", "")).strip()
        gt = _extract_gsm8k_final_answer(str(o.get("answer", "")))
        if not problem or not gt:
            continue
        sid = str(o.get("id", i)).strip()
        rows.append({"id": sid, "problem": problem, "ground_truth": gt})
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _choice_label(idx: int) -> str:
    return chr(ord("A") + idx)


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()


def load_mmlu_pro_hf_examples(
    path: Path,
    limit: Optional[int],
    mmlu_pro_config: str = "default",
    mmlu_pro_split: str = "test",
) -> List[Dict[str, Any]]:
    """
    Load MMLU-Pro from a HuggingFace-datasets style directory.
    Expected fields: question, options, answer_index (and/or answer)
    """
    rows: List[Dict[str, Any]] = []

    # 1) Prefer direct local parquet loading to avoid dataset_infos.json incompatibilities.
    split_key = mmlu_pro_split.strip()
    parquet_patterns = [
        f"{split_key}-*.parquet",
        f"{split_key}*.parquet",
    ]
    parquet_files: List[Path] = []
    for base in (path / "data", path):
        if not base.exists():
            continue
        for pat in parquet_patterns:
            parquet_files.extend(sorted(base.glob(pat)))
    # Deduplicate same parquet matched by multiple patterns (e.g. test-*.parquet
    # also matches test*.parquet), while preserving stable order.
    if parquet_files:
        parquet_files = list(dict.fromkeys(p.resolve() for p in parquet_files))
    if parquet_files:
        print(
            f"[eval] loading local MMLU-Pro split={split_key} from parquet files: {len(parquet_files)}",
            flush=True,
        )
        for pf_path in parquet_files:
            pf = pq.ParquetFile(pf_path)
            cols = ["question", "options", "answer", "answer_index", "category", "question_id"]
            existing_cols = [c for c in cols if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=512, columns=existing_cols):
                pyd = batch.to_pydict()
                n = len(next(iter(pyd.values()))) if pyd else 0
                for i in range(n):
                    o = {k: pyd[k][i] for k in pyd}
                    question = str(o.get("question", "")).strip()
                    raw_options = o.get("options", [])
                    if not isinstance(raw_options, list):
                        raw_options = list(raw_options) if raw_options is not None else []
                    options = [str(x).strip() for x in raw_options if str(x).strip()]
                    if not question or not options:
                        continue

                    answer_index_raw = o.get("answer_index", None)
                    answer_text_raw = str(o.get("answer", "")).strip()
                    gt_idx: Optional[int] = None
                    try:
                        if answer_index_raw is not None:
                            gt_idx = int(answer_index_raw)
                    except Exception:
                        gt_idx = None
                    if gt_idx is None and answer_text_raw:
                        ans_norm = _normalize_text(answer_text_raw)
                        for oi, opt in enumerate(options):
                            if _normalize_text(opt) == ans_norm:
                                gt_idx = oi
                                break
                    if gt_idx is None or gt_idx < 0 or gt_idx >= len(options):
                        continue

                    gt_letter = _choice_label(gt_idx)
                    option_lines = [f"{_choice_label(oi)}. {opt}" for oi, opt in enumerate(options)]
                    prompt_text = question + "\n\nOptions:\n" + "\n".join(option_lines)
                    sid = str(o.get("question_id", len(rows))).strip()
                    category = str(o.get("category", "")).strip()
                    rows.append(
                        {
                            "id": sid,
                            "problem": prompt_text,
                            "ground_truth": gt_letter,
                            "ground_truth_choice": gt_letter,
                            "ground_truth_text": options[gt_idx],
                            "options": options,
                            "category": category,
                            "eval_type": "mcq",
                        }
                    )
                    if limit is not None and len(rows) >= limit:
                        return rows
        if rows:
            return rows

    # 2) Fallback to datasets local directory/hub.
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Loading MMLU-Pro directory requires `datasets`. Install with: pip install datasets"
        ) from e

    ds = None
    load_err: Optional[Exception] = None
    try:
        ds = load_dataset(str(path), mmlu_pro_config, split=mmlu_pro_split)
    except Exception as e:
        load_err = e
        print(
            f"[warn] Failed to load local MMLU-Pro via datasets from {path}: {e}\n"
            "[warn] Falling back to HuggingFace hub dataset: TIGER-Lab/MMLU-Pro",
            flush=True,
        )
        ds = load_dataset("TIGER-Lab/MMLU-Pro", mmlu_pro_config, split=mmlu_pro_split)
    if ds is None:
        raise RuntimeError(
            f"Unable to load MMLU-Pro split={mmlu_pro_split} config={mmlu_pro_config} "
            f"from local path {path} or hub fallback."
        ) from load_err
    for i, o in enumerate(ds):
        question = str(o.get("question", "")).strip()
        raw_options = o.get("options", [])
        if not isinstance(raw_options, list):
            raw_options = list(raw_options) if raw_options is not None else []
        options = [str(x).strip() for x in raw_options if str(x).strip()]
        if not question or not options:
            continue

        answer_index_raw = o.get("answer_index", None)
        answer_text_raw = str(o.get("answer", "")).strip()

        gt_idx: Optional[int] = None
        try:
            if answer_index_raw is not None:
                gt_idx = int(answer_index_raw)
        except Exception:
            gt_idx = None
        if gt_idx is None and answer_text_raw:
            ans_norm = _normalize_text(answer_text_raw)
            for oi, opt in enumerate(options):
                if _normalize_text(opt) == ans_norm:
                    gt_idx = oi
                    break
        if gt_idx is None or gt_idx < 0 or gt_idx >= len(options):
            continue

        gt_letter = _choice_label(gt_idx)
        option_lines = [f"{_choice_label(oi)}. {opt}" for oi, opt in enumerate(options)]
        prompt_text = question + "\n\nOptions:\n" + "\n".join(option_lines)

        sid = str(o.get("question_id", i)).strip()
        category = str(o.get("category", "")).strip()
        rows.append(
            {
                "id": sid,
                "problem": prompt_text,
                "ground_truth": gt_letter,
                "ground_truth_choice": gt_letter,
                "ground_truth_text": options[gt_idx],
                "options": options,
                "category": category,
                "eval_type": "mcq",
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _parquet_loader_kind(path: Path) -> str:
    """Return 'dapo', 'amo_qa', or 'problem_answer' based on Parquet schema."""
    schema = pq.ParquetFile(path).schema_arrow
    names = set(schema.names)
    if "reward_model" in names:
        return "dapo"
    if "problem" in names and "answer" in names:
        return "problem_answer"
    if "question" in names and "answer" in names:
        # Keep GSM8K parquet files compatible with the generic problem+answer loader.
        return "problem_answer"
    if "prompt" in names and "answer" in names:
        pt = schema.field("prompt").type
        if pa.types.is_string(pt) or pa.types.is_large_string(pt):
            return "amo_qa"
    raise ValueError(
        f"Unsupported parquet schema in {path}; "
        f"need DAPO (reward_model+…), or string prompt+answer, or problem/question+answer. "
        f"Columns: {sorted(names)}"
    )


def load_amo_qa_parquet_examples(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    """AMO-Bench style: prompt (string), answer, optional question_id."""
    rows: List[Dict[str, Any]] = []
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    cols = ["prompt", "answer"]
    if "question_id" in names:
        cols = ["question_id", "prompt", "answer"]
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        if "question_id" in cols:
            qids = batch.column("question_id").to_pylist()
        else:
            qids = None
        prompts = batch.column("prompt").to_pylist()
        answers = batch.column("answer").to_pylist()
        for i, (pr, ans) in enumerate(zip(prompts, answers)):
            problem = str(pr).strip() if pr is not None else ""
            gt = str(ans).strip() if ans is not None else ""
            if not problem or not gt:
                continue
            if qids is not None:
                sid = str(qids[i]).strip() if qids[i] is not None else str(len(rows))
            else:
                sid = str(len(rows))
            rows.append({"id": sid, "problem": problem, "ground_truth": gt})
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def load_problem_answer_parquet_examples(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    """CMIMC / HMMT / BRUMO style: problem, answer; id from problem_idx or id if present."""
    rows: List[Dict[str, Any]] = []
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    text_col = "problem" if "problem" in names else "question" if "question" in names else None
    if text_col is None:
        raise ValueError(f"{path} has no 'problem' or 'question' column.")
    id_col: Optional[str] = None
    if "problem_idx" in names:
        id_col = "problem_idx"
    elif "id" in names:
        id_col = "id"
    cols = [text_col, "answer"]
    if id_col:
        cols = [id_col, text_col, "answer"]
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        if id_col:
            ids = batch.column(id_col).to_pylist()
        else:
            ids = None
        problems = batch.column(text_col).to_pylist()
        answers = batch.column("answer").to_pylist()
        for i, (pr, ans) in enumerate(zip(problems, answers)):
            problem = str(pr).strip() if pr is not None else ""
            gt = normalize_math_ground_truth(str(ans).strip() if ans is not None else "")
            if text_col == "question" and "####" in gt:
                gt = _extract_gsm8k_final_answer(gt)
            if not problem or not gt:
                continue
            if ids is not None:
                raw_id = ids[i]
                sid = str(raw_id).strip() if raw_id is not None else str(len(rows))
            else:
                sid = str(len(rows))
            rows.append({"id": sid, "problem": problem, "ground_truth": gt})
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def load_dapo_parquet_examples(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pf = pq.ParquetFile(path)
    cols = ["prompt", "reward_model", "extra_info"]
    for batch in pf.iter_batches(batch_size=512, columns=cols):
        prompts = batch.column("prompt").to_pylist()
        rewards = batch.column("reward_model").to_pylist()
        extras = batch.column("extra_info").to_pylist()
        for prompt_obj, reward_obj, extra_obj in zip(prompts, rewards, extras):
            problem = extract_user_prompt(prompt_obj)
            if not problem:
                continue
            gt = ""
            if isinstance(reward_obj, dict):
                gt = normalize_math_ground_truth(str(reward_obj.get("ground_truth", "")).strip())
            if not gt:
                continue
            sid = ""
            if isinstance(extra_obj, dict):
                sid = str(extra_obj.get("index", "")).strip()
            if not sid:
                sid = str(len(rows))
            rows.append({"id": sid, "problem": problem, "ground_truth": gt})
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def load_examples(
    path: Path,
    fmt: str,
    limit: Optional[int],
    gsm8k_config: str = "main",
    gsm8k_split: str = "test",
    mmlu_pro_config: str = "default",
    mmlu_pro_split: str = "test",
) -> List[Dict[str, Any]]:
    if fmt == "auto" and path.is_dir():
        has_hf_meta = (path / "dataset_infos.json").is_file() and (path / "README.md").is_file()
        has_gsm8k_meta = has_hf_meta and (path / "dataset_info.json").is_file()
        has_mmlu_pro_meta = has_hf_meta and "mmlu-pro" in path.name.lower()
        if has_gsm8k_meta:
            fmt = "gsm8k_hf"
        elif has_mmlu_pro_meta:
            fmt = "mmlu_pro_hf"
        else:
            raise ValueError(
                f"Cannot auto-detect format for directory {path}; set --data-format explicitly."
            )

    if fmt == "auto":
        suf = path.suffix.lower()
        if suf == ".jsonl":
            fmt = "jsonl"
        elif suf == ".parquet":
            kind = _parquet_loader_kind(path)
            fmt = {
                "dapo": "dapo_parquet",
                "amo_qa": "amo_qa_parquet",
                "problem_answer": "problem_answer_parquet",
            }[kind]
        else:
            raise ValueError(f"Cannot auto-detect format for suffix {suf}; set --data-format")
    if fmt == "jsonl":
        return load_jsonl_examples(path, limit)
    if fmt == "dapo_parquet":
        return load_dapo_parquet_examples(path, limit)
    if fmt == "amo_qa_parquet":
        return load_amo_qa_parquet_examples(path, limit)
    if fmt == "problem_answer_parquet":
        return load_problem_answer_parquet_examples(path, limit)
    if fmt == "gsm8k_hf":
        return load_gsm8k_hf_examples(path, limit, gsm8k_config=gsm8k_config, gsm8k_split=gsm8k_split)
    if fmt == "mmlu_pro_hf":
        return load_mmlu_pro_hf_examples(
            path,
            limit,
            mmlu_pro_config=mmlu_pro_config,
            mmlu_pro_split=mmlu_pro_split,
        )
    raise ValueError(f"Unknown --data-format: {fmt}")


def _adapter_dir_has_weights(d: Path) -> bool:
    return (d / "adapter_model.safetensors").is_file() or (d / "adapter_model.bin").is_file()


def _adapter_config_base_model(adapter_dir: Path) -> Optional[str]:
    p = adapter_dir / "adapter_config.json"
    if not p.is_file():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return meta.get("base_model_name_or_path") or meta.get("base_model_name")


def _infer_max_lora_rank_from_adapter(adapter_dir: Path, fallback: int) -> int:
    p = adapter_dir / "adapter_config.json"
    if not p.is_file():
        return fallback
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
        r = int(meta.get("r", fallback))
        return max(r, 1)
    except Exception:
        return fallback


def _is_peft_adapter_dir(d: Path) -> bool:
    return _adapter_dir_has_weights(d) and (d / "adapter_config.json").is_file()


def resolve_user_lora_dir(raw: Optional[str]) -> Optional[Path]:
    """
    Turn CLI LoRA path into the directory that actually holds adapter_config + weights.

    If ``raw`` is a training output parent (e.g. .../train) without weights at the root,
    use ``final/`` or ``lora_adapter/`` when present (matches train_preference.py layout).
    """
    if raw is None or not str(raw).strip():
        return None
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        return p
    if _is_peft_adapter_dir(p):
        return p
    for sub in ("final", "lora_adapter"):
        c = p / sub
        if _is_peft_adapter_dir(c):
            print(f"[eval] LoRA path {p} -> using adapter at {c}")
            return c
    return p


def resolve_vllm_base_and_lora(
    model_path: str,
    checkpoint_dir: Optional[str],
) -> tuple[str, Optional[Path]]:
    """
    Returns (vLLM base model path, optional LoRA adapter directory).

    If --checkpoint-dir / --lora-path is set, --model-path is the base model and that option is the adapter
    (after resolve_user_lora_dir, e.g. .../train -> .../train/final).
    Else if --model-path points to a PEFT adapter (adapter_config + adapter weights), base is read from
    adapter_config.json and the same path is the LoRA directory.
    """
    mp = Path(model_path).expanduser().resolve()
    ckpt_raw = checkpoint_dir.strip() if checkpoint_dir and str(checkpoint_dir).strip() else ""
    if ckpt_raw:
        return str(mp), Path(ckpt_raw).expanduser().resolve()
    if _is_peft_adapter_dir(mp):
        base_raw = _adapter_config_base_model(mp)
        if not base_raw:
            raise SystemExit(
                f"error: {mp} looks like a LoRA adapter but adapter_config.json "
                "has no base_model_name_or_path"
            )
        bpath = Path(base_raw)
        if bpath.exists():
            base_resolved = str(bpath.expanduser().resolve())
        else:
            base_resolved = base_raw
        return base_resolved, mp
    return str(mp), None


def build_llm(
    model_path: str,
    lora_path: Optional[str],
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    enforce_eager: bool,
    disable_custom_all_reduce: bool,
    max_lora_rank: int,
) -> Any:
    from vllm import LLM

    patch_vllm_ministral_tokenizer()
    from verl_rlsd.ministral_tokenizer import (
        patch_vllm_ministral_weight_files,
        patch_vllm_ministral_weight_loading,
    )

    patch_vllm_ministral_weight_files()
    patch_vllm_ministral_weight_loading()

    cfg: Dict[str, Any] = {
        "model": model_path,
        "tokenizer": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": "bfloat16",
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "disable_custom_all_reduce": disable_custom_all_reduce,
    }
    if enforce_eager:
        cfg["enforce_eager"] = True
    if lora_path:
        adapter_st = Path(lora_path) / "adapter_model.safetensors"
        adapter_bin = Path(lora_path) / "adapter_model.bin"
        if adapter_st.is_file() or adapter_bin.is_file():
            cfg["enable_lora"] = True
            cfg["max_lora_rank"] = max_lora_rank
            cfg["max_loras"] = 1
            cfg["max_cpu_loras"] = 1
        else:
            print(f"[warn] No adapter weights under {lora_path}; running without LoRA flags.")
            lora_path = None
    if is_ministral_model_path(model_path):
        cfg["hf_overrides"] = fix_ministral_hf_config
    return LLM(**cfg)


def default_data_root() -> Path:
    return Path(__file__).resolve().parent / "data"


# Aliases (lowercase, hyphen) -> path relative to data root
_DATASET_REL_PATH: Dict[str, str] = {}
for _aliases, _rel in (
    (("aime24",), "AIME24/test.parquet"),
    (("aime25",), "AIME25/test.parquet"),
    (("aime26",), "AIME26/test.parquet"),
    (("aime26-jsonl", "aime26-json"), "AIME26/aime2026.jsonl"),
    (("amc23",), "AMC23/test.parquet"),
    (("amo-bench", "amo_bench", "amobench"), "AMO-Bench/test.parquet"),
    (("brumo25",), "BRUMO25/test.parquet"),
    (("cmimc25",), "CMIMC25/test.parquet"),
    (
        ("dapo", "dapo-math", "dapo-math-17k", "dapo17k"),
        "preprocessed/dapo-math-17k.qwen3-4b.maxprompt1024.nothink.parquet",
    ),
    (("hmmt25", "hmmt-25"), "HMMT25/test.parquet"),
    (("math500", "math-500"), "MATH-500/test.parquet"),
    (("minerva",), "Minerva/test.parquet"),
    (("olympiad", "olympiad-bench", "olympiad_bench"), "Olympiad-Bench/test.parquet"),
    (("gsm8k", "gsm8k-main"), "gsm8k/main/test-00000-of-00001.parquet"),
    (("gsm8k-socratic",), "gsm8k/socratic/test-00000-of-00001.parquet"),
    (("mmlu-pro", "mmlu_pro", "mmlupro"), "mmlu-pro"),
):
    for _a in _aliases:
        _DATASET_REL_PATH[_a] = _rel


def normalize_dataset_key(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def resolve_dataset_path(name: str, data_root: Path, *, must_exist: bool = True) -> Path:
    key = normalize_dataset_key(name)
    rel = _DATASET_REL_PATH.get(key)
    if rel is None:
        known = ", ".join(sorted(set(_DATASET_REL_PATH.keys())))
        raise SystemExit(f"Unknown dataset {name!r}. Known aliases: {known}\n  (--data-root={data_root})")
    p = (data_root / rel).resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Dataset {name!r} -> expected path missing: {p}")
    return p


def extract_mcq_answer(text: str) -> Optional[str]:
    if not text:
        return None
    boxed = extract_boxed_answer(text)
    if boxed:
        m = re.search(r"\b([A-J])\b", boxed.upper())
        if m:
            return m.group(1)

    patterns = [
        r"(?:final answer|answer|correct option|option)\s*[:：]\s*\(?\s*([A-J])\s*\)?",
        r"\b([A-J])\b(?=\s*(?:\.|,|:|$))",
        r"\(([A-J])\)",
    ]
    up = text.upper()
    for pat in patterns:
        m = re.search(pat, up)
        if m:
            return m.group(1)
    return None


def _completion_token_count(
    text: str,
    token_ids: Optional[List[int]],
    tokenizer: Any,
) -> int:
    """Return completion length in tokens (prefer decoded text when vLLM token_ids look padded)."""
    text_count = len(tokenizer.encode(text, add_special_tokens=False)) if text else 0
    if not token_ids:
        return text_count
    ids_count = len(token_ids)
    # vLLM can return padded/incorrect token_ids for some models (e.g. Gemma 3).
    if text and ids_count > max(text_count + 128, text_count * 2):
        return text_count
    return ids_count


def _avg_output_tokens(rows: List[Dict[str, Any]]) -> float:
    counts: List[int] = []
    for r in rows:
        for g in r.get("generations", []):
            n = g.get("num_tokens")
            if n is not None:
                counts.append(int(n))
    return sum(counts) / len(counts) if counts else 0.0


def summarize_result_subset(
    rows: List[Dict[str, Any]],
    pass_at_k_list: List[int],
    gen_n: int,
) -> Dict[str, Any]:
    n_d = len(rows)
    pass_at_k: Dict[str, Dict[str, Any]] = {}
    for k in pass_at_k_list:
        c = sum(1 for r in rows if r.get("pass_at_k", {}).get(str(k)))
        pass_at_k[str(k)] = {
            "count": c,
            "total": n_d,
            "pct": 100.0 * c / n_d if n_d else 0.0,
        }
    maj = sum(1 for r in rows if r.get("majority_vote_correct"))
    total_sol = n_d * gen_n
    fmt = sum(sum(1 for g in r.get("generations", []) if g.get("formatted")) for r in rows)
    tot_correct = sum(r.get("num_correct", 0) for r in rows)
    avg1_pct = pass_at_k.get("1", {}).get("pct", 0.0)
    avg16_pct = 100.0 * tot_correct / total_sol if total_sol else 0.0
    return {
        "num_problems": n_d,
        "pass_at_k": pass_at_k,
        "avg1_pct": avg1_pct,
        "avg16_pct": avg16_pct,
        "majority_vote_pct": 100.0 * maj / n_d if n_d else 0.0,
        "average_correct_pct": 100.0 * tot_correct / total_sol if total_sol else 0.0,
        "format_rate_pct": 100.0 * fmt / total_sol if total_sol else 0.0,
        "avg_output_tokens_mean": _avg_output_tokens(rows),
    }


DEFAULT_TOKEN_BUDGETS = [1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_AP_LNE_K = 16
AP_LNE_SCALE = 1e5
AP_LNE_LOG_POWER = 4


def _format_token_budget_label(n: int) -> str:
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


def truncate_text_to_tokens(text: str, tokenizer: Any, max_tokens: int) -> str:
    if not text or max_tokens <= 0:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=False)


def _grade_generation_text(
    text: str,
    gt: str,
    *,
    eval_type: str = "boxed_math",
    gt_choice: str = "",
    relaxed: bool = False,
) -> bool:
    if eval_type == "mcq":
        pred = extract_mcq_answer(text)
        return bool(pred is not None and pred.upper() == gt_choice.upper().strip())
    pred = extract_math_answer(text, relaxed=relaxed)
    return grade_answer(pred, gt)


def _compute_token_budget_for_rows(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    pass_at_k_list: List[int],
    *,
    budgets: List[int],
    relaxed: bool,
) -> Dict[str, Dict[str, Any]]:
    n = len(rows)
    gen_n = len(rows[0]["generations"]) if rows else 0
    max_k = max(pass_at_k_list) if pass_at_k_list else 16
    by_budget: Dict[str, Dict[str, Any]] = {}
    for budget in budgets:
        pass_counts = {k: 0 for k in pass_at_k_list}
        total_correct = 0
        for row in rows:
            gt = row["ground_truth"]
            eval_type = str(row.get("eval_type", "boxed_math"))
            gt_choice = str(row.get("ground_truth_choice", gt))
            correct_flags: List[bool] = []
            for g in row.get("generations", []):
                truncated = truncate_text_to_tokens(g.get("full_generation", ""), tokenizer, budget)
                ok = _grade_generation_text(
                    truncated,
                    gt,
                    eval_type=eval_type,
                    gt_choice=gt_choice,
                    relaxed=relaxed,
                )
                correct_flags.append(ok)
                if ok:
                    total_correct += 1
            for k in pass_at_k_list:
                if any(correct_flags[:k]):
                    pass_counts[k] += 1
        total_sol = n * gen_n if gen_n else 0
        pass_at_k = {
            str(k): {
                "count": pass_counts[k],
                "total": n,
                "pct": 100.0 * pass_counts[k] / n if n else 0.0,
            }
            for k in pass_at_k_list
        }
        by_budget[str(budget)] = {
            "budget_tokens": budget,
            "label": _format_token_budget_label(budget),
            "num_problems": n,
            "pass_at_k": pass_at_k,
            "avg16_pct": 100.0 * total_correct / total_sol if total_sol else 0.0,
            f"pass{max_k}_pct": pass_at_k.get(str(max_k), {}).get("pct", 0.0),
        }
    return by_budget


def compute_token_budget_metrics(
    results: List[Dict[str, Any]],
    tokenizer: Any,
    pass_at_k_list: List[int],
    *,
    budgets: Optional[List[int]] = None,
    relaxed: bool = False,
) -> Dict[str, Any]:
    budgets = list(budgets or DEFAULT_TOKEN_BUDGETS)
    by_dataset: Dict[str, Dict[str, Any]] = {}
    tags: List[str] = []
    for row in results:
        tag = str(row.get("dataset_tag", "") or "__uncategorized__")
        if tag not in by_dataset:
            tags.append(tag)
            by_dataset[tag] = {"rows": []}
        by_dataset[tag]["rows"].append(row)
    for tag in tags:
        rows = by_dataset[tag]["rows"]
        by_dataset[tag] = {
            "num_problems": len(rows),
            "by_budget": _compute_token_budget_for_rows(
                rows,
                tokenizer,
                pass_at_k_list,
                budgets=budgets,
                relaxed=relaxed,
            ),
        }
    return {"budgets": budgets, "by_dataset": {tag: by_dataset[tag] for tag in tags}}


def format_token_budget_lines(metrics: Dict[str, Any], pass_at_k_list: List[int]) -> List[str]:
    lines = ["[TOKEN_BUDGET]"]
    max_k = max(pass_at_k_list) if pass_at_k_list else 16
    for tag, ds in metrics.get("by_dataset", {}).items():
        lines.append(f"[{tag}] n={ds['num_problems']}")
        for budget in metrics.get("budgets", []):
            m = ds["by_budget"][str(budget)]
            p = m["pass_at_k"][str(max_k)]
            lines.append(
                f"  @{m['label']}: Pass@{max_k}={p['pct']:.2f}% ({p['count']}/{m['num_problems']})  "
                f"Avg16={m['avg16_pct']:.2f}%"
            )
    return lines


def _harmonic_mean(a: float, b: float) -> float:
    if a + b <= 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def compute_ap_lne_metrics(
    results: List[Dict[str, Any]],
    *,
    k: int = DEFAULT_AP_LNE_K,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    tags: List[str] = []
    for row in results:
        tag = str(row.get("dataset_tag", "") or "__uncategorized__")
        if tag not in grouped:
            tags.append(tag)
            grouped[tag] = []
        grouped[tag].append(row)
    by_dataset: Dict[str, Dict[str, Any]] = {}
    for tag in tags:
        rows = grouped[tag]
        n = len(rows)
        pass_k_count = sum(1 for r in rows if r.get("pass_at_k", {}).get(str(k)))
        p_k = pass_k_count / n if n else 0.0
        gen_n = len(rows[0].get("generations", [])) if rows else 0
        total_correct = sum(int(r.get("num_correct", 0)) for r in rows)
        a_avg = total_correct / (n * gen_n) if n * gen_n else 0.0
        lengths: List[int] = []
        for row in rows:
            for gen in row.get("generations", []):
                tok = gen.get("num_tokens")
                if tok is not None:
                    lengths.append(int(tok))
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        hm = _harmonic_mean(a_avg, p_k)
        log_term = math.log2(1 + avg_len)
        denom = log_term ** AP_LNE_LOG_POWER
        ap_lne = (AP_LNE_SCALE * hm / denom) if denom > 0 else 0.0
        pass_at_k_pct = 100.0 * p_k
        avg16_pct = 100.0 * a_avg
        log_kl = math.log2(k * avg_len) if avg_len > 0 else 0.0
        pass_over_log_kl = (pass_at_k_pct / log_kl) if log_kl > 0 else 0.0
        avg_over_log_kl = (avg16_pct / log_kl) if log_kl > 0 else 0.0
        by_dataset[tag] = {
            "num_problems": n,
            "k": k,
            "pass_at_k": p_k,
            "pass_at_k_pct": pass_at_k_pct,
            "avg16": a_avg,
            "avg16_pct": avg16_pct,
            "avg_length": avg_len,
            "harmonic_mean": hm,
            "ap_lne": ap_lne,
            "log_kl": log_kl,
            "pass_over_log_kl": pass_over_log_kl,
            "avg_over_log_kl": avg_over_log_kl,
        }
    return {"k": k, "by_dataset": by_dataset}


def format_ap_lne_lines(metrics: Dict[str, Any]) -> List[str]:
    k = int(metrics.get("k", DEFAULT_AP_LNE_K))
    lines = [
        f"[AP-LNE@{k}]  1e5 * (2*A_avg*P_k/(A_avg+P_k)) / [log2(1+L)]^{AP_LNE_LOG_POWER}",
    ]
    for tag, ds in metrics.get("by_dataset", {}).items():
        lines.append(
            f"[{tag}] n={ds['num_problems']}  "
            f"P@{k}={ds['pass_at_k_pct']:.2f}%  Avg16={ds['avg16_pct']:.2f}%  L={ds['avg_length']:.1f}  "
            f"AP-LNE@{k}={ds['ap_lne']:.4f}"
        )
    lines.append(f"[PASS-AVG/LOG(kL)@{k}]  P@{k}/log2({k}*L), Avg@{k}/log2({k}*L)")
    for tag, ds in metrics.get("by_dataset", {}).items():
        lines.append(
            f"[{tag}] n={ds['num_problems']}  "
            f"P@{k}={ds['pass_at_k_pct']:.2f}%  Avg{k}={ds['avg16_pct']:.2f}%  L={ds['avg_length']:.1f}  "
            f"pass/log2({k}L)={ds['pass_over_log_kl']:.4f}  avg/log2({k}L)={ds['avg_over_log_kl']:.4f}"
        )
    return lines


def _remove_metric_block(text: str, marker: str) -> str:
    start = text.find(marker)
    if start < 0:
        return text
    end = text.find("\n" + "=" * 60, start)
    if end < 0:
        return text[:start].rstrip() + "\n"
    return text[:start] + text[end + 1 :]


def _remove_token_budget_block(text: str) -> str:
    return _remove_metric_block(text, "[TOKEN_BUDGET]")


def _remove_lne_block(text: str) -> str:
    for marker in ("[AP-LNE@", "[PASS-AVG/LOG(kL)@", "[LNE]"):
        text = _remove_metric_block(text, marker)
    return text


def _extract_metrics_path_from_out(text: str) -> Optional[Path]:
    match = re.search(
        r"(?:wrote final metrics ->|Wrote metrics)\s+(\S+\.metrics\.json)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return Path(match.group(1)).expanduser()


def append_lne_metrics_to_out(
    out_path: Path,
    results_path: Path,
    metrics_path: Path,
    *,
    ap_lne_k: int = DEFAULT_AP_LNE_K,
) -> None:
    with metrics_path.open(encoding="utf-8") as f:
        metrics_doc = json.load(f)
    with results_path.open(encoding="utf-8") as f:
        results_doc = json.load(f)
    results = results_doc.get("results", [])
    if not results:
        print(f"[postprocess] no results in {results_path}", flush=True)
        return
    ap_lne = compute_ap_lne_metrics(results, k=ap_lne_k)
    lines = format_ap_lne_lines(ap_lne)
    block = "\n".join(lines) + "\n"
    text = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
    text = _remove_lne_block(text)
    marker = "\n" + "=" * 60
    idx = text.rfind(marker)
    if idx >= 0:
        text = text[:idx] + "\n" + block + text[idx:]
    else:
        text = text.rstrip() + "\n\n" + block
    out_path.write_text(text, encoding="utf-8")
    print(f"[postprocess] appended LNE metrics to {out_path}", flush=True)


def _resolve_results_path(metrics_path: Path, results_path: Optional[Path] = None) -> Path:
    if results_path and results_path.is_file():
        return results_path
    candidate = metrics_path.parent / f"{metrics_path.stem.replace('.metrics', '')}.results.json"
    if candidate.is_file():
        return candidate
    stem = metrics_path.stem
    if stem.endswith(".metrics"):
        stem = stem[: -len(".metrics")]
    candidate = metrics_path.parent / f"{stem}.results.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"results json not found for metrics: {metrics_path}")


def append_postprocess_metrics_to_out(
    out_path: Path,
    results_path: Path,
    metrics_path: Path,
    *,
    budgets: Optional[List[int]] = None,
    ap_lne_k: int = DEFAULT_AP_LNE_K,
    replace: bool = True,
) -> None:
    if out_path.exists():
        text0 = out_path.read_text(encoding="utf-8")
        if not replace and "[TOKEN_BUDGET]" in text0 and "[AP-LNE@" in text0:
            print(f"[postprocess] skip (already present): {out_path}", flush=True)
            return
    with metrics_path.open(encoding="utf-8") as f:
        metrics_doc = json.load(f)
    with results_path.open(encoding="utf-8") as f:
        results_doc = json.load(f)
    results = results_doc.get("results", [])
    if not results:
        print(f"[postprocess] no results in {results_path}", flush=True)
        return
    model_path = metrics_doc.get("model_path") or metrics_doc.get("vllm_base_model_path", "")
    tokenizer = load_eval_tokenizer(model_path)
    pass_at_k_list = [int(x) for x in metrics_doc.get("pass_at_k_list", [1, 4, 8, 16])]
    relaxed = bool(metrics_doc.get("relaxed_answer_extraction", False))
    token_budget = compute_token_budget_metrics(
        results,
        tokenizer,
        pass_at_k_list,
        budgets=budgets,
        relaxed=relaxed,
    )
    ap_lne = compute_ap_lne_metrics(results, k=ap_lne_k)
    lines = format_token_budget_lines(token_budget, pass_at_k_list)
    lines.extend(format_ap_lne_lines(ap_lne))
    block = "\n".join(lines) + "\n"
    text = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
    text = _remove_lne_block(_remove_token_budget_block(text))
    marker = "\n" + "=" * 60
    idx = text.rfind(marker)
    if idx >= 0:
        text = text[:idx] + "\n" + block + text[idx:]
    else:
        text = text.rstrip() + "\n\n" + block
    out_path.write_text(text, encoding="utf-8")
    print(f"[postprocess] appended token budget + LNE to {out_path}", flush=True)
    for line in lines:
        print(line, flush=True)


def append_token_budget_to_out(
    out_path: Path,
    results_path: Path,
    metrics_path: Path,
    *,
    budgets: Optional[List[int]] = None,
    replace: bool = True,
) -> None:
    append_postprocess_metrics_to_out(
        out_path,
        results_path,
        metrics_path,
        budgets=budgets,
        replace=replace,
    )


def append_lne_to_out(
    out_path: Path,
    results_path: Path,
    metrics_path: Path,
    *,
    ap_lne_k: int = DEFAULT_AP_LNE_K,
    replace: bool = True,
) -> None:
    append_postprocess_metrics_to_out(
        out_path,
        results_path,
        metrics_path,
        ap_lne_k=ap_lne_k,
        replace=replace,
    )


def parse_pass_at_k(s: str) -> List[int]:
    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not parts:
        return [1]
    out = sorted(set(parts))
    if any(k < 1 for k in out):
        raise ValueError("--pass-at-k values must be positive integers")
    return out


def _eval_output_paths(base: Path) -> tuple[Path, Path]:
    """Derive streaming metrics/results paths from the configured --output-json base."""
    metrics_path = base.parent / f"{base.stem}.metrics.json"
    results_path = base.parent / f"{base.stem}.results.json"
    return metrics_path, results_path


def _load_resume_results(results_path: Path, gen_n: int) -> List[Dict[str, Any]]:
    """Load previously written per-problem rows that already have ``gen_n`` samples."""
    if not results_path.is_file():
        return []
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[eval] resume: failed to read {results_path}: {e}", flush=True)
        return []
    rows = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    kept: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        gens = r.get("generations") or []
        if not isinstance(gens, list) or len(gens) < gen_n:
            continue
        # Prefer exact match; if more samples exist, keep first gen_n for this run.
        if len(gens) > gen_n:
            r = dict(r)
            r["generations"] = gens[:gen_n]
            r["gen_n"] = gen_n
            flags = [bool(g.get("correct")) for g in r["generations"]]
            preds = [g.get("predicted_answer") for g in r["generations"]]
            formatted = [bool(g.get("formatted")) for g in r["generations"]]
            token_counts = [int(g.get("num_tokens") or 0) for g in r["generations"]]
            r["num_correct"] = sum(flags)
            r["pass_at_gen_n"] = bool(any(flags))
            r["avg_output_tokens"] = sum(token_counts) / len(token_counts) if token_counts else 0.0
            r["predicted_answer"] = preds[0] if preds else r.get("predicted_answer")
            r["full_generation"] = r["generations"][0].get("full_generation", "")
            r["correct"] = flags[0] if flags else False
            r["formatted"] = formatted[0] if formatted else False
            pass_at_k_problem = {}
            for k_str, ok in (r.get("pass_at_k") or {}).items():
                try:
                    k = int(k_str)
                except Exception:
                    continue
                pass_at_k_problem[str(k)] = bool(any(flags[:k]))
            r["pass_at_k"] = pass_at_k_problem
        kept.append(r)
    return kept


def _accumulate_result_stats(
    results: List[Dict[str, Any]],
    pass_at_k_list: List[int],
) -> tuple[Dict[int, int], int, int, int, int, int]:
    """Rebuild running counters from already-finished problem rows."""
    pass_at_k_counts: Dict[int, int] = {k: 0 for k in pass_at_k_list}
    formatted_total = 0
    total_solutions = 0
    total_correct = 0
    total_output_tokens = 0
    majority_correct = 0
    for r in results:
        gens = r.get("generations") or []
        for g in gens:
            total_solutions += 1
            if g.get("formatted"):
                formatted_total += 1
            if g.get("correct"):
                total_correct += 1
            total_output_tokens += int(g.get("num_tokens") or 0)
        for k in pass_at_k_list:
            if r.get("pass_at_k", {}).get(str(k)):
                pass_at_k_counts[k] += 1
        if r.get("majority_vote_correct"):
            majority_correct += 1
    return (
        pass_at_k_counts,
        formatted_total,
        total_solutions,
        total_correct,
        total_output_tokens,
        majority_correct,
    )


def _write_eval_checkpoints(
    *,
    metrics_path: Path,
    results_path: Path,
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
    processed: int,
    n_prompts: int,
) -> None:
    partial_only = processed < n_prompts
    disk_summary = {k: v for k, v in summary.items() if k != "results"}
    disk_summary["partial_only"] = partial_only
    disk_summary["results_count"] = len(results)
    disk_summary["results_json"] = str(results_path)
    metrics_path.write_text(
        json.dumps(disk_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    results_payload = {
        "partial_only": partial_only,
        "results_count": len(results),
        "num_problems_total": n_prompts,
        "metrics_json": str(metrics_path),
        "results": results,
    }
    results_path.write_text(
        json.dumps(results_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"[eval] wrote partial json: {processed}/{n_prompts} "
        f"metrics={metrics_path.name} results={results_path.name}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local math eval (vLLM + math_verify + boxed)")
    parser.add_argument("--model-path", type=str, default="", help="Base model dir (not needed for --list-datasets).")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Dataset name under --data-root (default: ./data). Repeat for multiple. "
            "Run with --list-datasets to see names (e.g. aime26, math500, dapo)."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="",
        help="Root directory for --dataset (default: <this_repo>/data).",
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=None,
        metavar="PATH",
        help="Explicit file path; repeat for multiple. Combined with resolved --dataset entries.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print known --dataset names and exit.",
    )
    parser.add_argument(
        "--data-format",
        type=str,
        default="auto",
        choices=[
            "auto",
            "jsonl",
            "dapo_parquet",
            "amo_qa_parquet",
            "problem_answer_parquet",
            "gsm8k_hf",
            "mmlu_pro_hf",
        ],
    )
    parser.add_argument(
        "--gsm8k-config",
        type=str,
        default="main",
        choices=["main", "socratic"],
        help="Used when --data-format=gsm8k_hf (or auto-detected gsm8k directory).",
    )
    parser.add_argument(
        "--gsm8k-split",
        type=str,
        default="test",
        help="Used when --data-format=gsm8k_hf (default: test).",
    )
    parser.add_argument(
        "--mmlu-pro-config",
        type=str,
        default="default",
        help="Used when --data-format=mmlu_pro_hf (default: default).",
    )
    parser.add_argument(
        "--mmlu-pro-split",
        type=str,
        default="test",
        help="Used when --data-format=mmlu_pro_hf (default: test).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        "--lora-path",
        dest="lora_arg",
        type=str,
        default=None,
        help=(
            "PEFT LoRA adapter directory (optional). Alias: --lora-path. "
            "May be train output root (.../train); then uses final/ or lora_adapter/ if weights are there."
        ),
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=0,
        help=(
            "vLLM max_lora_rank when using a LoRA adapter. "
            "0 = use r from adapter_config.json (fallback 64). "
            "If set below adapter r, it is raised automatically."
        ),
    )
    parser.add_argument("--output-json", type=str, default="", help="Summary JSON path (not needed for --list-datasets).")
    parser.add_argument("--num-samples", type=int, default=0, help="0 = use all rows")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="0 = auto (Gemma3=8192, thinking=38912, else 32768)",
    )
    parser.add_argument(
        "--fill-context",
        action="store_true",
        default=False,
        help=(
            "Set max_new_tokens = max_model_len - longest prompt length so prompt+completion "
            "fits in context (e.g. DeepSeek-Math 4096)."
        ),
    )
    parser.add_argument(
        "--relaxed-answer-extraction",
        action="store_true",
        default=False,
        help=(
            "For boxed_math: accept \\boxed{} or fallback phrases such as "
            "'The answer is: $...$' (DeepSeek-Math style)."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=-1.0,
        help="<0 = auto (Olmo=0.6; thinking=0.6; else 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=-1.0,
        help="<0 = auto (Olmo=0.95; thinking=0.95; else 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling cutoff; -1 disables. Qwen3 recommends 20; Olmo official settings omit top_k.",
    )
    parser.add_argument("--min-p", type=float, default=0.0, help="Set to 0 per Qwen3 official recommendation.")
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument(
        "--val-n",
        type=int,
        default=1,
        help="Samples per problem (vLLM n). Raised automatically to max(pass-at-k) if smaller.",
    )
    parser.add_argument(
        "--pass-at-k",
        type=str,
        default="1,4,8,16",
        help="Comma-separated k for pass@k (any correct in first k samples). Written to output JSON.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="0 = auto (max_prompt + max_new_tokens + 128)",
    )
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        default=False,
        help="Disable vLLM custom all-reduce and fall back to NCCL for tensor parallel inference.",
    )
    parser.add_argument(
        "--force-base-tokenizer",
        action="store_true",
        default=False,
        help=(
            "Always load tokenizer/chat template from --model-path (vLLM base), "
            "even when LoRA adapter dir contains tokenizer files."
        ),
    )
    parser.add_argument(
        "--generate-batch-size",
        type=int,
        default=0,
        help=(
            "Number of **problems** (prompts) per llm.generate() call. "
            "0 = one call with all prompts. "
            "Use 8–32 to cap concurrent prefill/decode batches (vLLM still schedules internally)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Continue from sibling *.results.json if present: skip problems that already have "
            "gen_n samples and only generate the remainder."
        ),
    )
    parser.add_argument(
        "--append-token-budget-to-out",
        type=str,
        default="",
        help="Post-process: append [TOKEN_BUDGET] metrics to this .out log file.",
    )
    parser.add_argument(
        "--token-budget-results-json",
        type=str,
        default="",
        help="Results JSON for --append-token-budget-to-out (default: sibling of metrics json).",
    )
    parser.add_argument(
        "--token-budget-metrics-json",
        type=str,
        default="",
        help="Metrics JSON for --append-token-budget-to-out.",
    )
    args = parser.parse_args()

    if args.append_token_budget_to_out:
        out_path = Path(args.append_token_budget_to_out).expanduser().resolve()
        metrics_path = Path(args.token_budget_metrics_json or "").expanduser().resolve()
        results_path = Path(args.token_budget_results_json or "").expanduser().resolve()
        if not metrics_path.is_file():
            raise SystemExit(f"--token-budget-metrics-json not found: {metrics_path}")
        results_path = _resolve_results_path(
            metrics_path,
            results_path if results_path.is_file() else None,
        )
        append_postprocess_metrics_to_out(out_path, results_path, metrics_path)
        return

    if args.list_datasets:
        root = Path(args.data_root).expanduser().resolve() if args.data_root.strip() else default_data_root()
        print(f"data_root: {root}\n")
        by_rel: Dict[str, List[str]] = {}
        for alias, rel in _DATASET_REL_PATH.items():
            by_rel.setdefault(rel, []).append(alias)
        for rel in sorted(by_rel.keys()):
            aliases = ", ".join(sorted(set(by_rel[rel])))
            p = root / rel
            ok = "ok" if p.exists() else "MISSING"
            print(f"  [{ok}] {rel}")
            print(f"       names: {aliases}")
        raise SystemExit(0)

    if not args.model_path.strip():
        raise SystemExit("error: --model-path is required (unless using --list-datasets)")
    if not args.output_json.strip():
        raise SystemExit("error: --output-json is required (unless using --list-datasets)")

    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root.strip()
        else default_data_root()
    )

    hf_dir_formats = {"gsm8k_hf", "mmlu_pro_hf"}
    load_queue: List[tuple[str, Optional[str]]] = []
    if args.dataset:
        for dn in args.dataset:
            p = resolve_dataset_path(
                dn,
                data_root,
                must_exist=args.data_format not in hf_dir_formats,
            )
            load_queue.append((str(p), normalize_dataset_key(dn)))
    if args.data_path:
        for raw in args.data_path:
            load_queue.append((raw, None))
    if not load_queue:
        raise SystemExit(
            "error: provide at least one --dataset and/or --data-path "
            "(or run with --list-datasets)"
        )

    pass_at_k_list = parse_pass_at_k(args.pass_at_k)
    max_k = max(pass_at_k_list)
    gen_n = max(args.val_n, max_k)
    if gen_n != args.val_n:
        print(f"[eval] val-n {args.val_n} < max(pass-at-k)={max_k}; generating n={gen_n} samples per problem.")

    if not _HAS_MATH_VERIFY:
        print("[warn] math_verify not installed; grading falls back to normalized string equality.")
        print("       pip install math-verify")

    limit = args.num_samples if args.num_samples > 0 else None
    resolved_paths: List[Path] = []
    examples: List[Dict[str, Any]] = []
    tag_counts: Dict[str, int] = {}
    for raw, tag_override in load_queue:
        data_path = Path(raw).expanduser().resolve()
        if args.data_format in hf_dir_formats:
            if not data_path.exists():
                if args.data_format == "mmlu_pro_hf":
                    print(
                        f"[warn] MMLU-Pro local dir missing ({data_path}); "
                        "will load from HuggingFace hub (TIGER-Lab/MMLU-Pro).",
                        flush=True,
                    )
                else:
                    raise FileNotFoundError(data_path)
        else:
            if not data_path.is_file():
                raise FileNotFoundError(data_path)
        resolved_paths.append(data_path)
        if args.data_format == "gsm8k_hf":
            batch = load_gsm8k_hf_examples(data_path, limit, gsm8k_config=args.gsm8k_config, gsm8k_split=args.gsm8k_split)
        elif args.data_format == "mmlu_pro_hf":
            batch = load_mmlu_pro_hf_examples(
                data_path,
                limit,
                mmlu_pro_config=args.mmlu_pro_config,
                mmlu_pro_split=args.mmlu_pro_split,
            )
        else:
            batch = load_examples(
                data_path,
                args.data_format,
                limit,
                gsm8k_config=args.gsm8k_config,
                gsm8k_split=args.gsm8k_split,
                mmlu_pro_config=args.mmlu_pro_config,
                mmlu_pro_split=args.mmlu_pro_split,
            )
        base_tag = tag_override if tag_override is not None else data_path.stem
        tag_counts[base_tag] = tag_counts.get(base_tag, 0) + 1
        tag = base_tag if tag_counts[base_tag] == 1 else f"{base_tag}_{tag_counts[base_tag]}"
        for ex in batch:
            ex["dataset_tag"] = tag
            ex["dataset_path"] = str(data_path)
        examples.extend(batch)
        print(f"[eval] +{len(batch)} problems from {data_path} (tag={tag})")

    if not examples:
        raise RuntimeError("No examples loaded; check --data-path and format.")

    lora_dir_cli = resolve_user_lora_dir(args.lora_arg)
    vllm_model_path, lora_dir = resolve_vllm_base_and_lora(
        args.model_path,
        str(lora_dir_cli) if lora_dir_cli is not None else None,
    )
    lora_dir_str = str(lora_dir) if lora_dir is not None else None
    is_gemma3 = _is_gemma3_model(vllm_model_path)
    is_olmo = is_olmo_model_path(vllm_model_path)
    enable_thinking = bool(args.enable_thinking) and not is_gemma3 and not is_olmo
    if is_gemma3 and args.enable_thinking:
        print(
            "[eval] Gemma 3 has no enable_thinking chat-template switch; using thinking=False",
            flush=True,
        )
    if is_olmo and args.enable_thinking:
        print(
            "[eval] Olmo-3-Instruct has no thinking mode; using official instruct settings",
            flush=True,
        )

    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens > 0
        else (38912 if enable_thinking else (8192 if is_gemma3 else 32768))
    )
    if args.temperature >= 0:
        temperature = args.temperature
    elif is_olmo or enable_thinking:
        temperature = 0.6
    else:
        temperature = 0.7
    if args.top_p >= 0:
        top_p = args.top_p
    elif is_olmo or enable_thinking:
        top_p = 0.95
    else:
        top_p = 0.8
    top_k = args.top_k
    min_p = max(args.min_p, 0.0)
    presence_penalty = args.presence_penalty

    print(f"[eval] total {len(examples)} problems from {len(resolved_paths)} file(s)")
    print(f"[eval] math_verify={'yes' if _HAS_MATH_VERIFY else 'no'}, thinking={enable_thinking}")

    tokenizer_src = vllm_model_path
    if (
        not args.force_base_tokenizer
        and lora_dir is not None
        and (lora_dir / "tokenizer_config.json").is_file()
    ):
        tokenizer_src = str(lora_dir.resolve())
    print(f"[eval] tokenizer_source={tokenizer_src}")
    tokenizer = load_eval_tokenizer(tokenizer_src)
    maybe_install_olmo_chat_template(tokenizer)

    all_prompts: List[str] = []
    for ex in examples:
        eval_type = str(ex.get("eval_type", "boxed_math"))
        user_suffix = _math_user_suffix(eval_type, is_gemma3)
        messages = [{"role": "user", "content": ex["problem"] + user_suffix}]
        all_prompts.append(_apply_chat_prompt(tokenizer, messages, enable_thinking))

    prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in all_prompts]
    max_prompt_tokens = max(prompt_lens) if prompt_lens else 0

    max_model_len = args.max_model_len
    if args.fill_context:
        if max_model_len <= 0:
            raise ValueError(
                "--fill-context requires --max-model-len > 0 so generation can fill remaining context."
            )
        max_new_tokens = max(1, max_model_len - max_prompt_tokens)
        print(
            f"[eval] fill-context: max_prompt_tokens={max_prompt_tokens}, "
            f"max_model_len={max_model_len} -> max_new_tokens={max_new_tokens}",
            flush=True,
        )
    elif max_model_len <= 0:
        max_model_len = max_prompt_tokens + max_new_tokens + 128
        print(
            f"[eval] auto max_model_len={max_model_len} "
            f"(max_prompt={max_prompt_tokens} + max_new_tokens={max_new_tokens} + 128)",
            flush=True,
        )
    elif max_model_len < max_prompt_tokens + max_new_tokens:
        raise ValueError(
            f"max_model_len={max_model_len} is smaller than prompt+generation budget "
            f"({max_prompt_tokens}+{max_new_tokens})."
        )

    max_lora_rank_cli = args.max_lora_rank
    if lora_dir is not None and _adapter_dir_has_weights(lora_dir):
        inferred_r = _infer_max_lora_rank_from_adapter(lora_dir, 64)
        if max_lora_rank_cli <= 0:
            max_lora_rank = inferred_r
        else:
            max_lora_rank = max(max_lora_rank_cli, inferred_r)
            if max_lora_rank_cli < inferred_r:
                print(
                    f"[warn] --max-lora-rank {max_lora_rank_cli} < adapter r={inferred_r}; "
                    f"using {max_lora_rank} for vLLM."
                )
        print(f"[eval] vLLM base={vllm_model_path} LoRA={lora_dir} max_lora_rank={max_lora_rank}")
    else:
        max_lora_rank = max_lora_rank_cli if max_lora_rank_cli > 0 else 64
        if lora_dir is not None and not _adapter_dir_has_weights(lora_dir):
            print(f"[warn] LoRA dir {lora_dir} has no adapter weights; eval runs base model only.")
        print(f"[eval] vLLM model={vllm_model_path}")

    cfg_max = max_seq_len_from_model_config(vllm_model_path)
    if cfg_max is not None and max_model_len > cfg_max:
        print(f"[eval] capping max_model_len {max_model_len} -> {cfg_max} (base model max_position_embeddings)")
        max_model_len = cfg_max
        if args.fill_context:
            max_new_tokens = max(1, max_model_len - max_prompt_tokens)
            print(
                f"[eval] fill-context (after max_model_len cap): max_new_tokens={max_new_tokens}",
                flush=True,
            )

    llm = build_llm(
        vllm_model_path,
        lora_dir_str,
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
        max_model_len,
        args.enforce_eager,
        args.disable_custom_all_reduce,
        max_lora_rank,
    )

    lora_request = None
    if lora_dir is not None and _adapter_dir_has_weights(lora_dir):
        try:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("eval_lora", 1, str(lora_dir.resolve()))
            print(f"[eval] LoRARequest -> {lora_dir}")
        except Exception as e:
            print(f"[warn] LoRA disabled: {e}")

    sp_kw: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "max_tokens": max_new_tokens,
        "n": gen_n,
        "seed": args.seed,
    }
    if top_k > 0:
        sp_kw["top_k"] = top_k
    if presence_penalty != 0.0:
        sp_kw["presence_penalty"] = presence_penalty

    from vllm import SamplingParams

    sampling_params = SamplingParams(**sp_kw)

    n_prompts_total = len(all_prompts)
    n_prompts = n_prompts_total
    gbs = args.generate_batch_size
    if gbs <= 0:
        gbs = n_prompts_total
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path, results_path = _eval_output_paths(out_path)

    results: List[Dict[str, Any]] = []
    pass_at_k_counts: Dict[int, int] = {k: 0 for k in pass_at_k_list}
    formatted_total = 0
    total_solutions = 0
    total_correct = 0
    total_output_tokens = 0
    majority_correct = 0
    processed = 0

    if args.resume:
        resumed = _load_resume_results(results_path, gen_n)
        if resumed:
            done_ids = {str(r.get("problem_id")) for r in resumed}
            keep_idx = [i for i, ex in enumerate(examples) if str(ex["id"]) not in done_ids]
            results = list(resumed)
            (
                pass_at_k_counts,
                formatted_total,
                total_solutions,
                total_correct,
                total_output_tokens,
                majority_correct,
            ) = _accumulate_result_stats(results, pass_at_k_list)
            processed = len(results)
            examples = [examples[i] for i in keep_idx]
            all_prompts = [all_prompts[i] for i in keep_idx]
            n_prompts = len(all_prompts)
            print(
                f"[eval] resume: loaded {processed} finished problems from {results_path.name}; "
                f"remaining {n_prompts}/{n_prompts_total} to generate",
                flush=True,
            )
        else:
            print(f"[eval] resume: no usable prior results at {results_path}", flush=True)

    print(
        f"[eval] generating {n_prompts} prompts x n={gen_n} "
        f"(pass-at-k={pass_at_k_list}, generate_batch_size={gbs}) ..."
    )

    use_inner_tqdm = n_prompts <= gbs
    for start in tqdm(
        range(0, n_prompts, gbs) if n_prompts > 0 else [],
        desc="prompt_batches",
        dynamic_ncols=True,
        disable=n_prompts <= gbs,
    ):
        end = min(start + gbs, n_prompts)
        chunk_prompts = all_prompts[start:end]
        chunk_examples = examples[start:end]
        if lora_request is not None:
            chunk_outputs = llm.generate(
                chunk_prompts,
                sampling_params,
                lora_request=lora_request,
                use_tqdm=use_inner_tqdm,
            )
        else:
            chunk_outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=use_inner_tqdm)
        if len(chunk_outputs) != len(chunk_examples):
            raise RuntimeError(
                f"expected {len(chunk_examples)} vLLM outputs in batch, got {len(chunk_outputs)}"
            )

        for ex, output in zip(chunk_examples, chunk_outputs):
            gt = ex["ground_truth"]
            generations: List[str] = []
            preds: List[str] = []
            correct_flags: List[bool] = []
            formatted_flags: List[bool] = []
            token_counts: List[int] = []

            for o in output.outputs:
                gen = o.text
                n_tokens = _completion_token_count(gen, getattr(o, "token_ids", None), tokenizer)
                generations.append(gen)
                token_counts.append(n_tokens)
                total_output_tokens += n_tokens
                eval_type = str(ex.get("eval_type", "boxed_math"))
                if eval_type == "mcq":
                    pred = extract_mcq_answer(gen)
                    formatted = pred is not None
                    gt_choice = str(ex.get("ground_truth_choice", ex["ground_truth"])).upper().strip()
                    ok = bool(pred is not None and pred.upper() == gt_choice)
                else:
                    pred = extract_math_answer(gen, relaxed=args.relaxed_answer_extraction)
                    formatted = pred is not None
                    ok = grade_answer(pred, gt)
                if pred is None:
                    preds.append("[no answer]" if args.relaxed_answer_extraction else "[no boxed]")
                else:
                    preds.append(pred)
                correct_flags.append(ok)
                formatted_flags.append(formatted)
                total_solutions += 1
                if formatted:
                    formatted_total += 1
                if ok:
                    total_correct += 1

            pass_at_k_problem: Dict[str, bool] = {}
            for k in pass_at_k_list:
                ok_k = any(correct_flags[:k])
                pass_at_k_problem[str(k)] = ok_k
                if ok_k:
                    pass_at_k_counts[k] += 1

            maj_ok = False
            fpreds = [p for p, f in zip(preds, formatted_flags) if f]
            if fpreds:
                top = Counter(fpreds).most_common(1)[0][0]
                maj_ok = grade_answer(top, gt)
            if maj_ok:
                majority_correct += 1

            results.append(
                {
                    "dataset_tag": ex.get("dataset_tag", ""),
                    "dataset_path": ex.get("dataset_path", ""),
                    "category": ex.get("category", ""),
                    "problem_id": ex["id"],
                    "problem": ex["problem"],
                    "ground_truth": gt,
                    "gen_n": gen_n,
                    "pass_at_k": pass_at_k_problem,
                    "generations": [
                        {
                            "predicted_answer": p,
                            "full_generation": g,
                            "num_tokens": n,
                            "correct": c,
                            "formatted": f,
                        }
                        for p, g, n, c, f in zip(
                            preds, generations, token_counts, correct_flags, formatted_flags
                        )
                    ],
                    "avg_output_tokens": sum(token_counts) / len(token_counts) if token_counts else 0.0,
                    "num_correct": sum(correct_flags),
                    "pass_at_gen_n": bool(any(correct_flags)),
                    "majority_vote_correct": maj_ok,
                    "predicted_answer": preds[0],
                    "full_generation": generations[0],
                    "correct": correct_flags[0],
                    "formatted": formatted_flags[0],
                }
            )

        processed = len(results)
        pass_at_k_summary: Dict[str, Dict[str, Any]] = {}
        for k in pass_at_k_list:
            c = pass_at_k_counts[k]
            pass_at_k_summary[str(k)] = {
                "count": c,
                "total": processed,
                "pct": 100.0 * c / processed if processed else 0.0,
            }
        avg1_pct = pass_at_k_summary.get("1", {}).get("pct", 0.0)
        avg16_pct = 100.0 * total_correct / total_solutions if total_solutions else 0.0

        by_tag: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in results:
            by_tag[str(r.get("dataset_tag", ""))].append(r)

        metrics_by_dataset: Dict[str, Any] = {}
        for tag, sub in sorted(by_tag.items(), key=lambda x: x[0]):
            path0 = sub[0].get("dataset_path", "") if sub else ""
            metrics_by_dataset[tag] = {
                "dataset_path": path0,
                **summarize_result_subset(sub, pass_at_k_list, gen_n),
            }

        by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in results:
            cat = str(r.get("category", "")).strip()
            if not cat:
                cat = "__uncategorized__"
            by_category[cat].append(r)
        metrics_by_category: Dict[str, Any] = {}
        for cat, sub in sorted(by_category.items(), key=lambda x: x[0]):
            metrics_by_category[cat] = summarize_result_subset(sub, pass_at_k_list, gen_n)

        summary = {
            "model_path": args.model_path,
            "vllm_base_model_path": vllm_model_path,
            "checkpoint_dir": args.lora_arg,
            "lora_adapter_dir": lora_dir_str,
            "max_lora_rank": max_lora_rank,
            "data_root": str(data_root),
            "data_paths": [str(p) for p in resolved_paths],
            "dataset_args": list(args.dataset) if args.dataset else [],
            "data_format": args.data_format,
            "enable_thinking": enable_thinking,
            "fill_context": args.fill_context,
            "relaxed_answer_extraction": args.relaxed_answer_extraction,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "max_new_tokens": max_new_tokens,
            "val_n_requested": args.val_n,
            "gen_n": gen_n,
            "pass_at_k_list": pass_at_k_list,
            "pass_at_k": pass_at_k_summary,
            "avg1_pct": avg1_pct,
            "avg16_pct": avg16_pct,
            "metrics_by_dataset": metrics_by_dataset,
            "metrics_by_category": metrics_by_category,
            "num_problems": processed,
            "num_problems_total": n_prompts_total,
            "total_solutions": total_solutions,
            "average_correct_pct": 100.0 * total_correct / total_solutions if total_solutions else 0.0,
            "majority_vote_pct": 100.0 * majority_correct / processed if processed else 0.0,
            "format_rate_pct": 100.0 * formatted_total / total_solutions if total_solutions else 0.0,
            "avg_output_tokens_mean": total_output_tokens / total_solutions if total_solutions else 0.0,
            "math_verify": _HAS_MATH_VERIFY,
            "generate_batch_size": gbs if args.generate_batch_size > 0 else n_prompts_total,
            "generate_batch_size_requested": args.generate_batch_size,
            "streaming_write": True,
            "metrics_json": str(metrics_path),
            "results_json": str(results_path),
            "results": results,
        }
        _write_eval_checkpoints(
            metrics_path=metrics_path,
            results_path=results_path,
            summary=summary,
            results=results,
            processed=processed,
            n_prompts=n_prompts_total,
        )

    # If everything was resumed (nothing left to generate), still materialize final summary.
    if n_prompts == 0 and results:
        processed = len(results)
        pass_at_k_summary = {}
        for k in pass_at_k_list:
            c = pass_at_k_counts[k]
            pass_at_k_summary[str(k)] = {
                "count": c,
                "total": processed,
                "pct": 100.0 * c / processed if processed else 0.0,
            }
        by_tag = defaultdict(list)
        for r in results:
            by_tag[str(r.get("dataset_tag", ""))].append(r)
        metrics_by_dataset = {}
        for tag, sub in sorted(by_tag.items(), key=lambda x: x[0]):
            path0 = sub[0].get("dataset_path", "") if sub else ""
            metrics_by_dataset[tag] = {
                "dataset_path": path0,
                **summarize_result_subset(sub, pass_at_k_list, gen_n),
            }
        by_category = defaultdict(list)
        for r in results:
            cat = str(r.get("category", "")).strip() or "__uncategorized__"
            by_category[cat].append(r)
        metrics_by_category = {
            cat: summarize_result_subset(sub, pass_at_k_list, gen_n)
            for cat, sub in sorted(by_category.items(), key=lambda x: x[0])
        }
        summary = {
            "model_path": args.model_path,
            "vllm_base_model_path": vllm_model_path,
            "checkpoint_dir": args.lora_arg,
            "lora_adapter_dir": lora_dir_str,
            "max_lora_rank": max_lora_rank,
            "data_root": str(data_root),
            "data_paths": [str(p) for p in resolved_paths],
            "dataset_args": list(args.dataset) if args.dataset else [],
            "data_format": args.data_format,
            "enable_thinking": enable_thinking,
            "fill_context": args.fill_context,
            "relaxed_answer_extraction": args.relaxed_answer_extraction,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "max_new_tokens": max_new_tokens,
            "val_n_requested": args.val_n,
            "gen_n": gen_n,
            "pass_at_k_list": pass_at_k_list,
            "pass_at_k": pass_at_k_summary,
            "avg1_pct": pass_at_k_summary.get("1", {}).get("pct", 0.0),
            "avg16_pct": 100.0 * total_correct / total_solutions if total_solutions else 0.0,
            "metrics_by_dataset": metrics_by_dataset,
            "metrics_by_category": metrics_by_category,
            "num_problems": processed,
            "num_problems_total": n_prompts_total,
            "total_solutions": total_solutions,
            "average_correct_pct": 100.0 * total_correct / total_solutions if total_solutions else 0.0,
            "majority_vote_pct": 100.0 * majority_correct / processed if processed else 0.0,
            "format_rate_pct": 100.0 * formatted_total / total_solutions if total_solutions else 0.0,
            "avg_output_tokens_mean": total_output_tokens / total_solutions if total_solutions else 0.0,
            "math_verify": _HAS_MATH_VERIFY,
            "generate_batch_size": gbs if args.generate_batch_size > 0 else n_prompts_total,
            "generate_batch_size_requested": args.generate_batch_size,
            "streaming_write": True,
            "metrics_json": str(metrics_path),
            "results_json": str(results_path),
            "results": results,
        }

    n = len(results)
    pass_at_k_summary = summary["pass_at_k"]
    metrics_by_dataset = summary["metrics_by_dataset"]
    metrics_by_category = summary["metrics_by_category"]

    summary["partial_only"] = False
    summary["metrics_json"] = str(metrics_path)
    summary["results_json"] = str(results_path)
    token_budget = compute_token_budget_metrics(
        results,
        tokenizer,
        pass_at_k_list,
        relaxed=bool(args.relaxed_answer_extraction),
    )
    ap_lne_metrics = compute_ap_lne_metrics(results)
    summary["token_budget_metrics"] = token_budget
    summary["ap_lne_metrics"] = ap_lne_metrics
    _write_eval_checkpoints(
        metrics_path=metrics_path,
        results_path=results_path,
        summary=summary,
        results=results,
        processed=n,
        n_prompts=n_prompts_total,
    )
    print(f"[eval] wrote final metrics -> {metrics_path}", flush=True)
    print(f"[eval] wrote final results -> {results_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("[ALL] combined", flush=True)
    for k in pass_at_k_list:
        s = pass_at_k_summary[str(k)]
        print(f"  Pass@{k}: {s['pct']:.2f}% ({s['count']}/{n})", flush=True)
    print(f"  Avg1(one-shot hit rate): {summary['avg1_pct']:.2f}%", flush=True)
    print(f"  Avg16(overall correctness): {summary['avg16_pct']:.2f}%", flush=True)
    print(f"  Avg output tokens: {summary['avg_output_tokens_mean']:.1f}", flush=True)
    for tag, m in metrics_by_dataset.items():
        print(f"[{tag}] n={m['num_problems']}", flush=True)
        for k in pass_at_k_list:
            s = m["pass_at_k"][str(k)]
            print(f"  Pass@{k}: {s['pct']:.2f}% ({s['count']}/{m['num_problems']})", flush=True)
        print(f"  Avg1(one-shot hit rate): {m['avg1_pct']:.2f}%", flush=True)
        print(f"  Avg16(overall correctness): {m['avg16_pct']:.2f}%", flush=True)
        print(f"  Avg output tokens: {m['avg_output_tokens_mean']:.1f}", flush=True)
    print("[BY_CATEGORY]", flush=True)
    for cat, m in metrics_by_category.items():
        print(f"[{cat}] n={m['num_problems']}", flush=True)
        for k in pass_at_k_list:
            s = m["pass_at_k"][str(k)]
            print(f"  Pass@{k}: {s['pct']:.2f}% ({s['count']}/{m['num_problems']})", flush=True)
        print(f"  Avg1(one-shot hit rate): {m['avg1_pct']:.2f}%", flush=True)
        print(f"  Avg16(overall correctness): {m['avg16_pct']:.2f}%", flush=True)
        print(f"  Avg output tokens: {m['avg_output_tokens_mean']:.1f}", flush=True)
    print(f"Avg correct / sample: {summary['average_correct_pct']:.2f}%", flush=True)
    print(f"Majority vote: {summary['majority_vote_pct']:.2f}%", flush=True)
    print(f"Boxed format rate: {summary['format_rate_pct']:.2f}%", flush=True)
    print(f"Avg output tokens: {summary['avg_output_tokens_mean']:.1f}", flush=True)
    for line in format_token_budget_lines(token_budget, pass_at_k_list):
        print(line, flush=True)
    for line in format_ap_lne_lines(ap_lne_metrics):
        print(line, flush=True)
    print(f"Wrote metrics {metrics_path}", flush=True)
    print(f"Wrote results {results_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
