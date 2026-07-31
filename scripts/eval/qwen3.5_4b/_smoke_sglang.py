#!/usr/bin/env python3
"""GPU smoke: load Qwen3.5-4B with SGLang Engine and generate one reply."""
from __future__ import annotations

from transformers import AutoTokenizer
from sglang import Engine


def main() -> None:
    model = "/gpfs/share/home/2501210611/labShare/2501210611/model/qwen35_4b"
    print("[smoke] loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    msgs = [{"role": "user", "content": "What is 2+2? Reply with only the number."}]
    prompt = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    print("[smoke] prompt head:", repr(prompt[:200]))

    print("[smoke] launching Engine attention=triton ...")
    llm = Engine(
        model_path=model,
        tokenizer_path=model,
        trust_remote_code=True,
        attention_backend="triton",
        mem_fraction_static=0.80,
        tp_size=1,
        context_length=8192,
        reasoning_parser="qwen3",
        log_level="error",
    )
    sp = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": 128,
        "presence_penalty": 1.5,
    }
    print("[smoke] generating...")
    outs = llm.generate([prompt], sampling_params=sp)
    text = outs[0]["text"] if isinstance(outs[0], dict) else outs[0]
    print("[smoke] output:", str(text)[:1000])
    llm.shutdown()
    print("[smoke] SUCCESS")


if __name__ == "__main__":
    main()
