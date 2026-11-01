#!/usr/bin/env python3
"""Offline smoke: Falcon vs generic LCB prompt assembly (no GPU / no HF)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional


def _load_helpers():
    path = Path(__file__).resolve().parent / "eval_lcb_vllm_local.py"
    src = path.read_text(encoding="utf-8")
    start = src.index("# Official Falcon-H1R default system")
    end = src.index("\ndef build_prompt(")
    prompt_constants = SimpleNamespace(
        SYSTEM_MESSAGE_GENERIC=(
            "You are an expert Python programmer. You will be given a question "
            "(problem specification) and will generate a correct Python program "
            "that matches the specification and passes all tests."
        ),
        FORMATTING_MESSAGE_WITH_STARTER_CODE=(
            "You will use the following starter code to write the solution to the "
            "problem and enclose your code within delimiters."
        ),
        FORMATTING_WITHOUT_STARTER_CODE=(
            "Read the inputs from stdin solve the problem and write the answer to "
            "stdout (do not directly test on the sample inputs). Enclose your code "
            "within delimiters as follows. Ensure that when the python program runs, "
            "it reads the inputs, runs the algorithm and writes output to STDOUT."
        ),
    )
    ns: dict = {
        "re": re,
        "Optional": Optional,
        "Any": Any,
        "PromptConstants": prompt_constants,
        "CodeGenerationProblem": object,
    }
    exec(src[start:end], ns)
    # Confirm build_prompt Falcon branch still in file.
    assert "FALCON_SYSTEM_MESSAGE" in src[end : end + 1200]
    assert "thinking_friendly=True" in src[end : end + 1200]
    return ns, prompt_constants


def main() -> None:
    ns, prompt_constants = _load_helpers()
    is_falcon_model = ns["is_falcon_model"]
    build_user_prompt = ns["build_user_prompt"]
    falcon_sys = ns["FALCON_SYSTEM_MESSAGE"]

    assert is_falcon_model("/labShare/model/falcon-h1r-7b")
    assert is_falcon_model("ocr_c256_falcon_h1r_7b_ckpt100")
    assert not is_falcon_model("/labShare/model/qwen3-8b")

    prob = SimpleNamespace(question_content="Given n, print n+1.", starter_code="")
    falcon_u = build_user_prompt(prob, thinking_friendly=True)
    generic_u = build_user_prompt(prob, thinking_friendly=False)

    assert "```python" in falcon_u
    assert "one ```python code block" in falcon_u
    assert "NOT return anything except" not in falcon_u
    assert "NOT return anything except" in generic_u
    assert "You are Falcon" in falcon_sys
    assert "<think>" in falcon_sys

    falcon_p = (
        f"<|im_start|>system\n{falcon_sys}<|im_end|>\n"
        f"<|im_start|>user\n{falcon_u}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )
    assert falcon_p.startswith("<|im_start|>system\nYou are Falcon")
    assert "expert Python programmer" not in falcon_p.split("<|im_start|>user")[0]
    assert prompt_constants.SYSTEM_MESSAGE_GENERIC.startswith("You are an expert Python")

    print("[ok] falcon detection")
    print("[ok] thinking-friendly user keeps ```python, drops 'NOT return anything'")
    print("[ok] Falcon system retained (not LCB generic)")
    print("--- Falcon prompt preview ---")
    print(falcon_p[:700])
    print("...")
    print(falcon_p[-180:])
    print("[smoke_lcb_prompt] PASS")


if __name__ == "__main__":
    main()
