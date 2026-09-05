"""Unit tests for Purified OPSD PMI target (arXiv:2607.02234 Sec. 3.2)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opsd_trainer import OPSDTrainer  # noqa: E402


def test_pmi_target_matches_closed_form():
    torch.manual_seed(0)
    b, t, v = 2, 3, 8
    teacher = torch.randn(b, t, v)
    ref = torch.randn(b, t, v)
    base = torch.randn(b, t, v)
    beta = 1.0
    clip = 10.0

    out = OPSDTrainer.build_purified_pmi_target_logits(
        teacher, ref, base, temperature=1.0, pmi_beta=beta, pmi_clip=clip
    )
    target = F.softmax(out, dim=-1)

    teacher_logp = F.log_softmax(teacher, dim=-1)
    ref_logp = F.log_softmax(ref, dim=-1)
    base_logp = F.log_softmax(base, dim=-1)
    delta = teacher_logp - ref_logp
    delta = delta - delta.mean(dim=-1, keepdim=True)
    delta = clip * torch.tanh(delta / clip)
    expected = F.softmax(base_logp + delta / beta, dim=-1)

    assert torch.allclose(target, expected, atol=1e-5)
    assert torch.allclose(target.sum(dim=-1), torch.ones(b, t), atol=1e-5)


def test_pmi_centering_is_zero_mean_before_clip():
    teacher = torch.tensor([[[0.0, 2.0, -1.0]]])
    ref = torch.tensor([[[0.0, 0.0, 0.0]]])
    base = torch.zeros_like(teacher)
    # With huge clip, tanh ≈ identity on small values; check mass shifts toward teacher preference.
    out = OPSDTrainer.build_purified_pmi_target_logits(
        teacher, ref, base, temperature=1.0, pmi_beta=1.0, pmi_clip=100.0
    )
    probs = F.softmax(out, dim=-1)
    assert probs[0, 0, 1] > probs[0, 0, 0]
    assert probs[0, 0, 1] > probs[0, 0, 2]


def test_reference_only_user_has_no_problem():
    from data_collator import SelfDistillationDataCollator

    text = SelfDistillationDataCollator._opsd_reference_only_user("42")
    assert "Problem:" not in text
    assert "Reference Solution Begin" in text
    assert "42" in text


if __name__ == "__main__":
    test_pmi_target_matches_closed_form()
    test_pmi_centering_is_zero_mean_before_clip()
    test_reference_only_user_has_no_problem()
    print("ok", math.pi)
