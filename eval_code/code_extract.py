#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Think-mode-aware Python extraction for EvalPlus / LCB judges.

Default (strict): only ```python / ```py fences count. No fence → empty
(instruction-following failure). Copy the fence body verbatim. Prefer a
fence that defines the entrypoint over a trailing print/assert demo.

Instruct chat + nothink (Olmo Instruct/IT, Qwen3-*-Instruct, OCR instruct
ckpts) often continues a prompt-prefilled ```python block or dumps bare
``def`` / ``class Solution`` without a language tag. Use profile
``olmo_instruct`` for those families.

SFT think models (``sft_think``) often invent a descriptive function name
instead of MBPP's assert entrypoint (or mismatch case only). Same fence
rules as strict, then rename a single / case-matched top-level ``def`` to
``entrypoint`` so EvalPlus sanitize does not drop the solution.
"""

from __future__ import annotations

import ast
import re
import textwrap
from typing import List, Optional, Tuple

_SANITIZE_MAX_LINES = 2000

PROFILE_STRICT = "strict"
PROFILE_OLMO_INSTRUCT = "olmo_instruct"
PROFILE_SFT_THINK = "sft_think"

_DEF_LINE = re.compile(
    r"(?m)^(?P<indent>\s*)(?P<async>async\s+)?def\s+(?P<name>\w+)\s*\("
)

_PYTHON_FENCE_LANG = frozenset({"python", "py"})
_IMPL_HEAD = re.compile(
    r"^(?:"
    r"(?:async\s+)?(?:def|class)\s+"
    r"|from\s+\S+\s+import\b"
    r"|import\s+"
    r"|if\s+__name__\s*=="
    r"|[A-Za-z_][A-Za-z0-9_]*\s*="
    r"|#)"
)


def answer_region(text: str) -> str:
    s = text or ""
    idx = s.lower().rfind("</think>")
    if idx < 0:
        return s
    return s[idx + len("</think>") :]


def _syntax_ok(code: str) -> bool:
    try:
        ast.parse(code or "")
        return True
    except SyntaxError:
        return False


def _is_python_fence_lang(tag: str) -> bool:
    lang = (tag or "").strip().split()[0].lower() if (tag or "").strip() else ""
    return lang in _PYTHON_FENCE_LANG


def infer_extract_profile(
    *,
    model_path: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
    hint_path: Optional[str] = None,
) -> str:
    """Pick extract profile from checkpoint / output path.

    Think / base models typically emit ```python after ``</think>`` — keep strict.
    Instruct / IT chat (Olmo Instruct, Qwen3-*-Instruct, OCR instruct ckpts)
    continue a prompt-prefilled ```python block or dump bare ``def``.

    ``sft_think`` is never auto-selected here (OCR / base / Olmo stay unchanged).
    Pass ``--extract-profile sft_think`` explicitly from SFT judge jobs only.
    """
    del enable_thinking  # path is the source of truth; Instruct eval is nothink.
    blob = f"{model_path or ''} {hint_path or ''}".lower()
    has_instruct = bool(re.search(r"instruct|\bit\b|-it(?:/|_|\b)", blob))
    if has_instruct:
        return PROFILE_OLMO_INSTRUCT
    return PROFILE_STRICT


def _top_level_def_names(code: str) -> List[str]:
    """Top-level function names (AST when possible, else def-line regex)."""
    text = code or ""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [m.group("name") for m in _DEF_LINE.finditer(text)]
    names: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
    return names


def _rename_def(code: str, old: str, new: str) -> str:
    if not old or not new or old == new:
        return code
    return _DEF_LINE.sub(
        lambda m: (
            f"{m.group('indent')}{m.group('async') or ''}def {new}("
            if m.group("name") == old
            else m.group(0)
        ),
        code or "",
        count=1,
    )


def _align_entrypoint(code: str, entrypoint: Optional[str]) -> str:
    """Rename a mismatched top-level def to ``entrypoint`` when safe.

    Rules (conservative):
    - already has exact entrypoint → unchanged
    - case-only mismatch → rename that def
    - exactly one top-level def → rename it
    - otherwise leave unchanged (avoid guessing among helpers)
    """
    if not entrypoint or not (code or "").strip():
        return code or ""
    if _has_def_entry(code, entrypoint):
        return code
    defs = _top_level_def_names(code)
    if not defs:
        return code
    for name in defs:
        if name.lower() == entrypoint.lower():
            return _rename_def(code, name, entrypoint)
    if len(defs) == 1:
        return _rename_def(code, defs[0], entrypoint)
    return code


def _iter_fence_bodies(
    text: str, *, allow_unlabeled: bool = False
) -> List[Tuple[int, str, bool]]:
    """(start, body, unclosed) for ```python / ```py, optionally unlabeled ```."""
    out: List[Tuple[int, str, bool]] = []
    s = text or ""
    i = 0
    while True:
        start = s.find("```", i)
        if start < 0:
            break
        nl = s.find("\n", start + 3)
        if nl < 0:
            break
        tag = s[start + 3 : nl]
        close = s.find("```", nl + 1)
        if close < 0:
            if _accept_fence_tag(tag, allow_unlabeled=allow_unlabeled):
                body = s[nl + 1 :]
                if body.strip():
                    out.append((start, body, True))
            break
        if _accept_fence_tag(tag, allow_unlabeled=allow_unlabeled):
            body = s[nl + 1 : close]
            if body.strip():
                out.append((start, body, False))
        i = close + 3
    return out


def _accept_fence_tag(tag: str, *, allow_unlabeled: bool) -> bool:
    if _is_python_fence_lang(tag):
        return True
    if not allow_unlabeled:
        return False
    lang = (tag or "").strip().split()[0].lower() if (tag or "").strip() else ""
    return lang == ""


def _has_def_entry(code: str, entrypoint: Optional[str]) -> bool:
    if not entrypoint:
        return bool(re.search(r"(?m)^\s*(?:async\s+)?def\s+\w+\s*\(", code or ""))
    return (
        re.search(
            rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entrypoint)}\s*\(",
            code or "",
        )
        is not None
    )


def _has_impl(code: str) -> bool:
    return bool(
        re.search(r"(?m)^\s*(?:async\s+)?(?:def |class |import |from )", code or "")
    )


def _is_demo_or_assert_only(code: str) -> bool:
    lines = [
        ln.strip()
        for ln in (code or "").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines or _has_impl(code):
        return False
    return all(
        ln.startswith("assert ")
        or ln.startswith("print(")
        or ln.startswith(">>> ")
        for ln in lines
    )


def _normalize_fence(body: str) -> str:
    text = body.replace("\t", "    ")
    if text.startswith("\n"):
        text = text[1:]
    text = text.rstrip("\n")
    if _syntax_ok(text):
        return text
    nonempty = [ln for ln in text.splitlines() if ln.strip()]
    if nonempty and all(ln[:1] in " \t" for ln in nonempty):
        dedented = textwrap.dedent(text)
        if _syntax_ok(dedented):
            return dedented
    return text


def _trim_trailing_until_parse(code: str) -> str:
    if _syntax_ok(code):
        return code
    lines = code.splitlines()
    if len(lines) > _SANITIZE_MAX_LINES:
        lines = lines[:_SANITIZE_MAX_LINES]
    while lines:
        cand = "\n".join(lines).rstrip()
        if _syntax_ok(cand):
            return cand
        lines.pop()
    return ""


def _slice_from_first_code_line(text: str) -> str:
    lines = (text or "").splitlines()
    start: Optional[int] = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or s.startswith("```"):
            continue
        if _IMPL_HEAD.match(s):
            start = i
            break
    if start is None:
        return (text or "").strip()
    return "\n".join(lines[start:]).strip()


def _raw_python_candidate(text: str) -> str:
    body = _slice_from_first_code_line(answer_region(text))
    if not body:
        return ""
    norm = _normalize_fence(body)
    parsed = _trim_trailing_until_parse(norm)
    if _syntax_ok(parsed) and (_has_impl(parsed) or _IMPL_HEAD.match(parsed.lstrip())):
        return parsed
    if _syntax_ok(norm) and (_has_impl(norm) or _IMPL_HEAD.match(norm.lstrip())):
        return norm
    return ""


def extract_evalplus_candidate(
    solution: str,
    entrypoint: Optional[str] = None,
    profile: str = PROFILE_STRICT,
) -> str:
    text = solution or ""
    think_end = text.lower().rfind("</think>")
    ranked: List[Tuple[Tuple[int, ...], str]] = []
    # sft_think shares strict fence rules; rename is applied at the end.
    fence_profile = (
        PROFILE_OLMO_INSTRUCT if profile == PROFILE_OLMO_INSTRUCT else PROFILE_STRICT
    )

    def add(pos: int, body: str) -> None:
        if not (body or "").strip():
            return
        after = 1 if (think_end < 0 or pos > think_end) else 0
        has_entry = 1 if _has_def_entry(body, entrypoint) else 0
        parses = 1 if _syntax_ok(body) else 0
        has_impl = 1 if _has_impl(body) else 0
        not_demo = 0 if _is_demo_or_assert_only(body) else 1
        ranked.append(((has_entry, parses, has_impl, not_demo, after, pos), body))

    def _finish(body: str) -> str:
        if profile == PROFILE_SFT_THINK:
            return _align_entrypoint(body, entrypoint)
        return body

    for pos, body, unclosed in _iter_fence_bodies(text):
        norm = _normalize_fence(body)
        if unclosed:
            parsed = _trim_trailing_until_parse(norm)
            if parsed:
                add(pos, parsed)
        else:
            add(pos, norm)

    if ranked:
        ranked.sort(key=lambda x: x[0], reverse=True)
        for key, body in ranked:
            if key[0] and key[1]:
                return _finish(body)
        for key, body in ranked:
            if key[1] and key[2] and key[3]:
                return _finish(body)
        best = ranked[0][1]
        if _syntax_ok(best) or _has_def_entry(best, entrypoint):
            return _finish(best)

    if fence_profile != PROFILE_OLMO_INSTRUCT:
        return ""

    for pos, body, unclosed in _iter_fence_bodies(text, allow_unlabeled=True):
        norm = _normalize_fence(body)
        cand = _trim_trailing_until_parse(norm) if unclosed else norm
        add(pos, cand if cand else norm)
    if ranked:
        ranked.sort(key=lambda x: x[0], reverse=True)
        for key, body in ranked:
            if key[0] and key[1]:
                return _finish(body)
        for key, body in ranked:
            if key[1] and key[2] and key[3]:
                return _finish(body)

    raw = _raw_python_candidate(text)
    if raw and (not entrypoint or _has_def_entry(raw, entrypoint) or _has_impl(raw)):
        return _finish(raw)
    return ""


def extract_lcb_candidate(
    text: str, profile: str = PROFILE_STRICT
) -> str:
    """Last valid ```python block; empty if the model never opened one.

    ``olmo_instruct`` also accepts unlabeled fences and bare stdin/leetcode code.
    ``sft_think`` keeps the same fence rules as strict (no entrypoint rename).
    """
    s = text or ""
    region = answer_region(s)
    allow = profile == PROFILE_OLMO_INSTRUCT
    fences = _iter_fence_bodies(region, allow_unlabeled=allow) or _iter_fence_bodies(
        s, allow_unlabeled=allow
    )
    parsed: List[str] = []
    for _pos, body, unclosed in fences:
        norm = _normalize_fence(body)
        cand = _trim_trailing_until_parse(norm) if unclosed else norm
        if _syntax_ok(cand):
            parsed.append(cand)
        elif cand.strip() and not unclosed:
            parsed.append(norm)
    if parsed:
        return parsed[-1]
    if profile != PROFILE_OLMO_INSTRUCT:
        return ""
    return _raw_python_candidate(s)
