from __future__ import annotations

from typing import Any

import torch


class SelfDistillationDataCollator:
    """Build Qwen3 student/privileged-teacher prompts for an OPSD batch."""

    MODES = {"correct", "pi", "instruction"}

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 9216,
        max_prompt_length: int = 1024,
        privilege_mode: str = "correct",
        student_thinking: bool = False,
        teacher_thinking: bool = False,
        **_: Any,
    ) -> None:
        if privilege_mode not in self.MODES:
            raise ValueError(f"privilege_mode must be one of {sorted(self.MODES)}")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.max_prompt_length = int(max_prompt_length)
        self.privilege_mode = privilege_mode
        self.student_thinking = bool(student_thinking)
        self.teacher_thinking = bool(teacher_thinking)
        self.reason_first = False
        self.tokenizer.padding_side = "right"

    def format_prompts(self, feature: dict[str, Any]) -> tuple[str, str]:
        problem = str(feature["problem"]).strip()
        solution = str(feature.get("solution", "")).strip()

        if self.privilege_mode == "instruction":
            student_instruction = (
                "Give a concise solution with only the essential reasoning, and put the final answer "
                "within \\boxed{}."
            )
            teacher_user = (
                f"Problem: {problem}\n\n"
                "Give a detailed, rigorous solution. Explain every important derivation, check the result, "
                "and put the final answer within \\boxed{}."
            )
        else:
            student_instruction = (
                "Please reason step by step and put the final answer within \\boxed{}."
            )
            privileged = solution if self.privilege_mode == "correct" else "π"
            label = "verified answer" if self.privilege_mode == "correct" else "privileged answer"
            teacher_user = (
                f"Problem: {problem}\n\n"
                f"Here is the {label}:\n"
                "=== Privileged Information Begin ===\n"
                f"{privileged}\n"
                "=== Privileged Information End ===\n\n"
                "After understanding the privileged information, solve the problem using your own reasoning. "
                "Please reason step by step and put the final answer within \\boxed{}."
            )

        student_user = f"Problem: {problem}\n\n{student_instruction}"
        student_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": student_user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.student_thinking,
        )
        teacher_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": teacher_user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.teacher_thinking,
        )
        return student_prompt, teacher_prompt

    def prompt_lengths(self, feature: dict[str, Any]) -> tuple[int, int]:
        student, teacher = self.format_prompts(feature)
        return (
            len(self.tokenizer(student, add_special_tokens=False)["input_ids"]),
            len(self.tokenizer(teacher, add_special_tokens=False)["input_ids"]),
        )

    def fits(self, feature: dict[str, Any]) -> bool:
        student_len, teacher_len = self.prompt_lengths(feature)
        return student_len <= self.max_prompt_length and teacher_len <= self.max_prompt_length

    def _encode(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        no_pad = self.tokenizer(prompts, padding=False, truncation=False, add_special_tokens=False)
        lengths = [len(ids) for ids in no_pad["input_ids"]]
        if max(lengths) > self.max_prompt_length:
            raise ValueError(
                f"prompt exceeds max_prompt_length={self.max_prompt_length}; "
                "filter the dataset with collator.fits before training"
            )
        encoded = self.tokenizer(
            prompts,
            padding="longest",
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"], lengths

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        pairs = [self.format_prompts(feature) for feature in features]
        student_ids, student_mask, student_lengths = self._encode([x[0] for x in pairs])
        teacher_ids, teacher_mask, teacher_lengths = self._encode([x[1] for x in pairs])
        return {
            "student_prompts": student_ids,
            "student_prompt_attention_mask": student_mask,
            "student_prompt_length": student_ids.shape[1],
            "student_prompt_lengths_per_example": torch.tensor(student_lengths),
            "teacher_prompts": teacher_ids,
            "teacher_prompt_attention_mask": teacher_mask,
            "teacher_prompt_length": teacher_ids.shape[1],
            "teacher_prompt_lengths_per_example": torch.tensor(teacher_lengths),
        }
