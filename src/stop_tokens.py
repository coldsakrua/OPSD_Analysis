"""Shared stop-token helpers for Qwen / Qwen3.5 chat models."""

from __future__ import annotations

from typing import Any


def resolve_stop_token_ids(tokenizer: Any) -> list[int]:
    """Return EOS / chat-end ids: <|im_end|>, <|endoftext|>, plus tokenizer.eos.

    SFT/base checkpoints often only list <|endoftext|> in generation_config;
    chat rollouts still need <|im_end|> or decoding never stops.
    """
    ids: list[int] = []
    unk = getattr(tokenizer, "unk_token_id", None)

    def _add(tid: Any) -> None:
        if not isinstance(tid, int) or tid < 0:
            return
        if unk is not None and tid == unk:
            return
        if tid not in ids:
            ids.append(tid)

    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        _add(eos)
    elif isinstance(eos, list):
        for x in eos:
            _add(x)

    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    for tok in ("<|im_end|>", "<|endoftext|>"):
        if isinstance(vocab, dict) and tok in vocab:
            _add(vocab[tok])
            continue
        try:
            _add(tokenizer.convert_tokens_to_ids(tok))
        except Exception:
            pass
    return ids
