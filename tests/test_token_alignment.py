"""Static checks for student/teacher response alignment and pad-gap packing."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opsd_sequence import (  # noqa: E402
    assert_response_token_alignment,
    detect_pad_gap_before_response,
    left_align_prompt_completion,
    prompt_region_has_right_padding,
)


PAD = 0


def _right_padded_prompts() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Row0 real len 2, row1 real len 4 (batch max).
    prompts = torch.tensor(
        [
            [11, 12, PAD, PAD],
            [21, 22, 23, 24],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([2, 4], dtype=torch.long)
    return prompts, lengths, prompts


def test_naive_right_pad_concat_has_pad_gap() -> None:
    prompts, _, _ = _right_padded_prompts()
    completions = torch.tensor([[31, 32, PAD], [41, 42, 43]], dtype=torch.long)
    comp_attn = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
    naive_ids = torch.cat([prompts, completions], dim=1)
    naive_attn = torch.cat([(prompts != PAD).long(), comp_attn], dim=1)
    prompt_width = prompts.shape[1]
    assert detect_pad_gap_before_response(naive_attn, prompt_width) == 1
    assert bool(prompt_region_has_right_padding(naive_attn, prompt_width)[0])
    assert not bool(prompt_region_has_right_padding(naive_attn, prompt_width)[1])
    # First response logit index would be prompt_width-1 == pad for short row.
    assert naive_ids[0, prompt_width - 1].item() == PAD


def test_left_align_removes_pad_gap_and_keeps_response() -> None:
    prompts, lengths, _ = _right_padded_prompts()
    completions = torch.tensor([[31, 32, PAD], [41, 42, 43]], dtype=torch.long)
    comp_attn = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
    ids, attn, labels, prompt_width = left_align_prompt_completion(
        prompts, lengths, completions, comp_attn, pad_token_id=PAD
    )
    assert prompt_width == 4
    assert detect_pad_gap_before_response(attn, prompt_width) == 0
    # Last prompt position is always a real token.
    assert ids[0, prompt_width - 1].item() == 12
    assert ids[1, prompt_width - 1].item() == 24
    # Response starts immediately after prompt_width.
    assert torch.equal(ids[0, prompt_width : prompt_width + 2], torch.tensor([31, 32]))
    assert torch.equal(ids[1, prompt_width : prompt_width + 3], torch.tensor([41, 42, 43]))
    # Labels only on real completion tokens.
    assert torch.equal(labels[0, prompt_width : prompt_width + 2], torch.tensor([31, 32]))
    assert labels[0, prompt_width + 2].item() == -100
    assert (labels[:, :prompt_width] == -100).all()


def test_student_teacher_response_alignment() -> None:
    student_prompts = torch.tensor([[1, 2, PAD], [3, 4, 5]], dtype=torch.long)
    teacher_prompts = torch.tensor(
        [
            [9, 9, 9, 9, PAD, PAD],
            [8, 8, 8, 8, 8, 7],
        ],
        dtype=torch.long,
    )
    student_lengths = torch.tensor([2, 3], dtype=torch.long)
    teacher_lengths = torch.tensor([4, 6], dtype=torch.long)
    completions = torch.tensor([[10, 11, PAD], [12, 13, 14]], dtype=torch.long)
    comp_attn = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)

    student_ids, student_attn, labels, sp = left_align_prompt_completion(
        student_prompts, student_lengths, completions, comp_attn, PAD
    )
    teacher_ids, teacher_attn, _, tp = left_align_prompt_completion(
        teacher_prompts, teacher_lengths, completions, comp_attn, PAD
    )
    assert_response_token_alignment(
        student_ids,
        teacher_ids,
        sp,
        tp,
        student_attn[:, sp:],
        teacher_attn[:, tp:],
    )
    # Causal-shift slices used in compute_loss must be the same response length.
    assert student_ids[:, sp:].shape == teacher_ids[:, tp:].shape
    assert labels[:, sp:].shape == student_ids[:, sp:].shape


def test_misaligned_response_raises() -> None:
    student = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    teacher = torch.tensor([[9, 9, 3, 5]], dtype=torch.long)
    mask = torch.tensor([[1, 1]], dtype=torch.long)
    try:
        assert_response_token_alignment(student, teacher, 2, 2, mask)
    except AssertionError as exc:
        assert "misaligned" in str(exc)
    else:
        raise AssertionError("expected misalignment to raise")


if __name__ == "__main__":
    test_naive_right_pad_concat_has_pad_gap()
    test_left_align_removes_pad_gap_and_keeps_response()
    test_student_teacher_response_alignment()
    test_misaligned_response_raises()
    print("test_token_alignment: all passed")
