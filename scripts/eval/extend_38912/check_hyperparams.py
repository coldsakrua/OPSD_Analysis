#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check Qwen3 think eval scripts / metrics against official hyperparams."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

OFFICIAL = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "max_new_tokens_math": 38912,
    "max_model_len": 40960,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3]),
    )
    args = parser.parse_args()
    base = Path(args.base_dir).resolve()
    scripts_root = base / "scripts" / "eval"
    bad_scripts = []
    ok_thinking = []
    for sh in sorted(scripts_root.rglob("*think*.sh")):
        text = sh.read_text(encoding="utf-8", errors="ignore")
        rel = str(sh.relative_to(base))
        # Skip Thinking-2507 and non-Qwen3 long-context stacks that intentionally differ.
        if "qwen3_4b_thinking" in rel or "qwen3.5" in rel or "qwen35" in rel:
            continue
        if "deepseek" in rel or "falcon" in rel or "olmo" in rel or "mimo" in rel:
            continue
        m_new = re.search(r"--max-new-tokens\s+(\d+|\"[^\"]+\"|\$\{[^}]+\})", text)
        m_len = re.search(r"--max-model-len\s+(\d+|\"[^\"]+\"|\$\{[^}]+\})", text)
        # Hardcoded 32768 is the bug we fix.
        if re.search(r"--max-new-tokens\s+32768\b", text):
            bad_scripts.append((rel, "max-new-tokens=32768 (want 38912 for math think)"))
        elif re.search(r"--max-new-tokens\s+38912\b", text) or "MAX_NEW_TOKENS" in text:
            ok_thinking.append(rel)
        if m_len and re.search(r"--max-model-len\s+40960\b", text):
            pass
        elif m_len and "40960" not in (m_len.group(1) if m_len else ""):
            # Only flag if clearly not 40960 and not a variable for thinking models already skipped.
            if "32768" in text and "max-model-len" in text:
                bad_scripts.append((rel, f"max-model-len looks non-40960: {m_len.group(1)}"))

    print("=== Official targets ===")
    for k, v in OFFICIAL.items():
        print(f"  {k}={v}")
    print(f"\n=== Think scripts still on 32768: {len(bad_scripts)} ===")
    for rel, reason in bad_scripts[:50]:
        print(f"  BAD {rel}: {reason}")
    if len(bad_scripts) > 50:
        print(f"  ... and {len(bad_scripts) - 50} more")
    print(f"\n=== Think scripts already 38912 / MAX_NEW_TOKENS: {len(ok_thinking)} ===")

    # Metrics extended check
    eval_outputs = base / "eval_outputs"
    remaining = []
    extended = []
    for mp in eval_outputs.glob("*/*think*.metrics.json"):
        if "nothink" in mp.name.lower():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if int(m.get("extended_to") or 0) >= 38912 or int(m.get("max_new_tokens") or 0) >= 38912:
            extended.append(str(mp.relative_to(base)))
        elif int(m.get("max_new_tokens") or 0) == 32768:
            remaining.append(str(mp.relative_to(base)))
    print(f"\n=== Metrics max_new>=38912 or extended_to: {len(extended)} ===")
    print(f"=== Metrics still max_new=32768: {len(remaining)} ===")


if __name__ == "__main__":
    main()
