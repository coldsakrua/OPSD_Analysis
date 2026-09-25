"""Presets for excluding token positions from OPSD distillation loss."""

from __future__ import annotations

import re

# Think-mode (st_tt): deliberation / hedge tokens with high JSD or negative advantage.
REASONING_WORDS = [
    "wait",
    "hmm",
    "but",
    "maybe",
    "probably",
    "might",
    "seems",
    "actually",
    "mistake",
    "wrong",
    "check",
    "verify",
    "let",
    "so",
    "we",
]

# Instruct-mode (snt_tnt): layout, math/markdown, and final-answer scaffolding.
STRUCTURE_TOKENS = [
    "\n\n",
    ".\n\n",
    ":\n\n",
    "---\n\n",
    " $",
    "$",
    "$$",
    " **",
    "**",
    " \\(",
    "\\(",
    " \\)",
    "\\)",
    " \\",
    "\\",
    " Final",
    "Final",
    " Answer",
    "Answer",
]

_REASONING_WORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in REASONING_WORDS) + r")\b",
    re.IGNORECASE,
)

DISTILL_EXCLUDE_PRESETS = frozenset({"reasoning", "structure"})


def char_ranges_for_preset(text: str, preset: str) -> list[tuple[int, int]]:
    """Return [start, end) character spans to exclude from distillation loss."""
    if preset == "reasoning":
        return [(m.start(), m.end()) for m in _REASONING_WORD_RE.finditer(text)]
    if preset == "structure":
        ranges: list[tuple[int, int]] = []
        for needle in STRUCTURE_TOKENS:
            if not needle:
                continue
            start = 0
            while True:
                idx = text.find(needle, start)
                if idx < 0:
                    break
                ranges.append((idx, idx + len(needle)))
                start = idx + 1
        return ranges
    raise ValueError(f"unknown distill exclude preset: {preset}")
