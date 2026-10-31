#!/usr/bin/env python3
"""Print code-extraction diagnostics from a generations JSON (LCB / EvalPlus)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from code_extract import (
    answer_region,
    extract_evalplus_candidate,
    extract_lcb_candidate,
    infer_extract_profile,
    _syntax_ok,
)


def _has_python_fence(text: str) -> bool:
    return bool(re.search(r"```(?:python|py)\b", text or "", flags=re.I))


def _has_think_close(text: str) -> bool:
    return "</think>" in (text or "").lower()


def _preview(text: str, n: int = 500) -> str:
    s = (text or "").replace("\r\n", "\n")
    if len(s) <= n:
        return s
    return s[: n // 2] + "\n...\n" + s[-n // 2 :]


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    doc = json.loads(path.read_text())
    if isinstance(doc, list):
        return doc
    raise SystemExit(f"Expected list generations JSON: {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generations-json", required=True)
    p.add_argument("--dataset", required=True, choices=["lcb_v6", "humanevalplus", "mbppplus"])
    p.add_argument("--model-path", default="")
    p.add_argument("--extract-profile", default="auto")
    p.add_argument("--max-show", type=int, default=3)
    args = p.parse_args()

    path = Path(args.generations_json)
    rows = _load_rows(path)
    profile = args.extract_profile
    if profile in ("", "auto"):
        profile = infer_extract_profile(model_path=args.model_path, hint_path=str(path))

    print(f"[smoke-extract] file={path}")
    print(f"[smoke-extract] dataset={args.dataset} n_rows={len(rows)} profile={profile}")

    ok_n = empty_n = no_fence_n = no_think_close_n = syntax_fail_n = 0
    for i, row in enumerate(rows):
        tid = row.get("task_id") or row.get("question_id") or row.get("id") or f"row{i}"
        gens: List[str] = row.get("output_list") or []
        raw = gens[0] if gens else ""
        entry = row.get("entry_point") or row.get("entrypoint")

        if args.dataset == "lcb_v6":
            code = extract_lcb_candidate(raw, profile=profile)
        else:
            code = extract_evalplus_candidate(raw, entrypoint=entry, profile=profile)

        has_fence = _has_python_fence(raw)
        has_close = _has_think_close(raw)
        region = answer_region(raw)
        syn = bool(code) and _syntax_ok(code)

        if not has_fence:
            no_fence_n += 1
        if not has_close:
            no_think_close_n += 1
        if not code:
            empty_n += 1
            status = "EMPTY"
        elif not syn:
            syntax_fail_n += 1
            status = "BAD_SYNTAX"
        else:
            ok_n += 1
            status = "OK"

        if i < args.max_show:
            print("=" * 72)
            print(f"[{status}] {tid} raw_chars={len(raw)} extracted_chars={len(code)}")
            print(
                f"  fence_python={has_fence} </think>={has_close} "
                f"answer_region_chars={len(region)} syntax_ok={syn}"
            )
            print("--- raw (trim) ---")
            print(_preview(raw))
            print("--- extracted ---")
            print(_preview(code) if code else "<EMPTY>")

    print("=" * 72)
    print(
        f"[smoke-extract] SUMMARY ok={ok_n}/{len(rows)} empty={empty_n} "
        f"no_python_fence={no_fence_n} no_think_close={no_think_close_n} "
        f"bad_syntax={syntax_fail_n} profile={profile}"
    )
    if empty_n == len(rows):
        print("[smoke-extract] VERDICT=FAIL (all extractions empty)")
    elif empty_n or syntax_fail_n:
        print("[smoke-extract] VERDICT=WARN (some extraction issues)")
    else:
        print("[smoke-extract] VERDICT=PASS")
    # Always exit 0 so multi-dataset smoke jobs continue after a WARN/FAIL.


if __name__ == "__main__":
    main()
