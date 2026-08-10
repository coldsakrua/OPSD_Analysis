"""Helpers for packing student/teacher rollout sequences without pad gaps.

Collator right-pads prompts to the batch max. Concatenating completions after that
padding yields ``[real_prompt | PAD... | completion]``, so the first response logit
is taken from a pad position when using a batch-level ``prompt_len``.

Left-aligning the real prompt to the right edge of a shared prompt width fixes
prompt→response continuity while keeping a single batch ``prompt_length`` for
causal LM slicing: ``logits[:, prompt_len - 1 : -1]``.
"""

from __future__ import annotations

import torch


def left_align_prompt_completion(
    prompts: torch.Tensor,
    prompt_lengths: torch.Tensor,
    completions: torch.Tensor,
    completion_attention: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack ``[PAD...][real_prompt][completion][PAD...]`` per row.

    Args:
        prompts: ``[B, P]`` right-padded prompt ids (collator layout).
        prompt_lengths: ``[B]`` unpadded prompt lengths.
        completions: ``[B, C]`` right-padded completion ids.
        completion_attention: ``[B, C]`` 1 for real completion tokens.
        pad_token_id: padding id.

    Returns:
        ``input_ids``, ``attention_mask``, ``labels`` (prompt / pad → -100),
        and ``prompt_width`` (= ``P``).
    """
    if prompts.ndim != 2 or completions.ndim != 2:
        raise ValueError("prompts and completions must be rank-2")
    if prompts.shape[0] != completions.shape[0]:
        raise ValueError("batch size mismatch between prompts and completions")
    batch, prompt_width = prompts.shape
    max_c = completions.shape[1]
    device = prompts.device
    dtype = prompts.dtype

    lengths = prompt_lengths.to(device=device, dtype=torch.long).view(-1)
    if lengths.numel() != batch:
        raise ValueError("prompt_lengths must have shape [B]")
    if bool((lengths < 0).any() or (lengths > prompt_width).any()):
        raise ValueError("prompt_lengths out of range for prompts width")

    input_ids = prompts.new_full((batch, prompt_width + max_c), int(pad_token_id))
    attention_mask = torch.zeros_like(input_ids)
    labels = input_ids.new_full(input_ids.shape, -100)

    for i in range(batch):
        lp = int(lengths[i].item())
        if lp > 0:
            input_ids[i, prompt_width - lp : prompt_width] = prompts[i, :lp]
            attention_mask[i, prompt_width - lp : prompt_width] = 1
        lc = int(completion_attention[i].sum().item())
        if lc > 0:
            input_ids[i, prompt_width : prompt_width + lc] = completions[i, :lc]
            attention_mask[i, prompt_width : prompt_width + lc] = 1
            labels[i, prompt_width : prompt_width + lc] = completions[i, :lc]

    return input_ids, attention_mask, labels, prompt_width


def prompt_region_has_right_padding(attention_mask: torch.Tensor, prompt_width: int) -> torch.Tensor:
    """Return per-row bool: prompt region has a 1→0 transition (right-pad before response)."""
    prompt_attn = attention_mask[:, :prompt_width].to(dtype=torch.long)
    if prompt_width <= 1:
        return torch.zeros(attention_mask.shape[0], dtype=torch.bool, device=attention_mask.device)
    return ((prompt_attn[:, :-1] == 1) & (prompt_attn[:, 1:] == 0)).any(dim=1)


def assert_response_token_alignment(
    student_input_ids: torch.Tensor,
    teacher_input_ids: torch.Tensor,
    student_prompt_len: int,
    teacher_prompt_len: int,
    student_response_mask: torch.Tensor,
    teacher_response_mask: torch.Tensor | None = None,
) -> None:
    """Assert student/teacher share the same non-pad response token ids."""
    student_resp = student_input_ids[:, student_prompt_len:]
    teacher_resp = teacher_input_ids[:, teacher_prompt_len:]
    if student_resp.shape != teacher_resp.shape:
        raise AssertionError(
            f"response shape mismatch: student {tuple(student_resp.shape)} "
            f"vs teacher {tuple(teacher_resp.shape)}"
        )
    mask = student_response_mask.bool()
    if teacher_response_mask is not None:
        tmask = teacher_response_mask.bool()
        if not torch.equal(mask, tmask):
            raise AssertionError("student/teacher response attention masks differ")
    if mask.shape != student_resp.shape:
        raise AssertionError("response mask shape mismatch")
    if not torch.equal(student_resp[mask], teacher_resp[mask]):
        raise AssertionError("student/teacher response token ids are misaligned")


def detect_pad_gap_before_response(
    attention_mask: torch.Tensor,
    prompt_width: int,
) -> int:
    """Count rows where right-padding sits between real prompt tokens and response."""
    bad = prompt_region_has_right_padding(attention_mask, prompt_width)
    return int(bad.sum().item())
