from __future__ import annotations

from pathlib import Path
from typing import Any

OLMO_DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful function-calling AI assistant. "
    "You do not currently have access to any functions. <functions></functions>"
)

# Fallback for Olmo *base* / Instruct when chat_template.jinja is missing.
# Think models ship chat_template.jinja that ends generation prompt with "<think>".
OLMO_CHAT_TEMPLATE = """{%- set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 -%}
{%- if not has_system -%}
<|im_start|>system
""" + OLMO_DEFAULT_SYSTEM_MESSAGE + """
<|im_end|>
{%- endif -%}
{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
<|im_start|>system
{{ message['content'] }}
<|im_end|>
{%- elif message['role'] == 'user' -%}
<|im_start|>user
{{ message['content'] }}
<|im_end|>
{%- elif message['role'] == 'assistant' -%}
<|im_start|>assistant
{{ message['content'] }}
<|im_end|>
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{%- endif -%}"""


def is_olmo_tokenizer(tokenizer: Any) -> bool:
    if tokenizer is None:
        return False
    name_or_path = str(getattr(tokenizer, "name_or_path", "") or "").lower()
    if "olmo" in name_or_path:
        return True
    config = getattr(tokenizer, "config", None)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    if model_type in {"olmo3", "olmo2", "olmo"}:
        return True
    init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
    init_name = str(init_kwargs.get("name_or_path", "") or "").lower()
    return "olmo" in init_name


def is_olmo_model_path(model_path: str | None) -> bool:
    return "olmo" in str(model_path or "").lower()


def is_olmo_think_model_path(model_path: str | None) -> bool:
    """True for Olmo-3-*Think* (path / HF id containing 'think')."""
    p = str(model_path or "").lower()
    return "olmo" in p and "think" in p


def is_olmo_instruct_model_path(model_path: str | None) -> bool:
    """True for Olmo Instruct / IT; False for Think."""
    if not is_olmo_model_path(model_path):
        return False
    return not is_olmo_think_model_path(model_path)


def install_olmo_chat_template(tokenizer: Any) -> None:
    """Install ChatML-style chat template for Olmo base models."""
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return
    if getattr(tokenizer, "chat_template", None):
        return
    tokenizer.chat_template = OLMO_CHAT_TEMPLATE
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and inner is not tokenizer and not getattr(inner, "chat_template", None):
        inner.chat_template = OLMO_CHAT_TEMPLATE


def load_olmo_chat_template(model_path: str | None = None) -> str:
    """Return Jinja chat template: prefer model-dir chat_template.jinja, else Instruct fallback."""
    if model_path:
        jinja = Path(model_path).expanduser() / "chat_template.jinja"
        if jinja.is_file():
            return jinja.read_text(encoding="utf-8")
    return OLMO_CHAT_TEMPLATE


def maybe_install_olmo_chat_template(tokenizer: Any, model_path: str | None = None) -> None:
    """
    Ensure Olmo tokenizer has a chat template.

    Prefer the model's shipped ``chat_template.jinja`` (Think prepends ``<think>``;
    Instruct does not). Only fall back to the hard-coded Instruct template when neither
    tokenizer.chat_template nor a jinja file is available.
    """
    if not is_olmo_tokenizer(tokenizer) and not is_olmo_model_path(model_path):
        return
    if getattr(tokenizer, "chat_template", None):
        return
    path = model_path or str(getattr(tokenizer, "name_or_path", "") or "")
    template = load_olmo_chat_template(path if path else None)
    tokenizer.chat_template = template
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and inner is not tokenizer and not getattr(inner, "chat_template", None):
        inner.chat_template = template
