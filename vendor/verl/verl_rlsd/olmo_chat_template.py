from __future__ import annotations

from typing import Any

OLMO_DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful function-calling AI assistant. "
    "You do not currently have access to any functions. <functions></functions>"
)

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
    """Return Jinja chat template string for Olmo base/instruct models."""
    _ = model_path
    return OLMO_CHAT_TEMPLATE


def is_olmo_model_path(model_path: str | None) -> bool:
    return "olmo" in str(model_path or "").lower()


def maybe_install_olmo_chat_template(tokenizer: Any) -> None:
    if is_olmo_tokenizer(tokenizer):
        install_olmo_chat_template(tokenizer)
