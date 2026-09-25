"""Build per-position masks that drop tokens from OPSD / JSD distillation loss."""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from distill_mask_presets import char_ranges_for_preset


def _token_char_spans(tokenizer: PreTrainedTokenizerBase, token_ids: list[int]) -> list[tuple[int, int]]:
    if not token_ids:
        return []
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    spans: list[tuple[int, int]] = []
    cursor = 0
    for tid in token_ids:
        piece = tokenizer.decode(
            [tid],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if piece and text[cursor : cursor + len(piece)] == piece:
            start, end = cursor, cursor + len(piece)
            cursor = end
        elif piece:
            idx = text.find(piece, cursor)
            if idx < 0:
                start, end = cursor, cursor
            else:
                start, end = idx, idx + len(piece)
                cursor = end
        else:
            start, end = cursor, cursor
        spans.append((start, end))
    return spans


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def exclude_flags_for_completion(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: list[int],
    preset: str,
    *,
    mask_eos: bool = True,
) -> list[bool]:
    """True = exclude this completion token from distillation loss."""
    if not token_ids:
        return []
    spans = _token_char_spans(tokenizer, token_ids)
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    bad_ranges = char_ranges_for_preset(text, preset)
    flags = [
        any(_spans_overlap(tok_span, bad) for bad in bad_ranges) for tok_span in spans
    ]
    if mask_eos and preset == "structure":
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            for i, tid in enumerate(token_ids):
                if tid == eos_id:
                    flags[i] = True
    return flags


def build_distill_exclude_mask(
    tokenizer: PreTrainedTokenizerBase,
    sampled_token_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    preset: str,
) -> torch.Tensor:
    """Bool [batch, seq]: positions to exclude from distillation (True = no loss)."""
    exclude = torch.zeros_like(valid_mask, dtype=torch.bool)
    batch_size = int(sampled_token_ids.shape[0])
    for i in range(batch_size):
        valid_idx = valid_mask[i].nonzero(as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        ids = sampled_token_ids[i, valid_idx].tolist()
        row_flags = exclude_flags_for_completion(tokenizer, ids, preset)
        for j, pos in enumerate(valid_idx.tolist()):
            if j < len(row_flags) and row_flags[j]:
                exclude[i, pos] = True
    return exclude
