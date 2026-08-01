from __future__ import annotations

from typing import Any

import torch


# Matches https://github.com/siyan-zhao/OPSD data_collator.py (non reason_first).
OFFICIAL_TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)

# No-GT variant: keep the transition intent, but never mention a reference solution.
NO_GT_TRANSITION_PROMPT = (
    "\n\nMake sure you truly understand the problem. Using your own words and independent "
    "reasoning, derive the final answer. Think step by step, explore different approaches, "
    "and don't be afraid to backtrack or reconsider if something doesn't work out:\n"
)

PRIVILEGE_FIELDS = {"solution", "answer", "none"}

# No-GT teacher prefixes (no answer / solution leakage).
ENCOURAGE_PREFIX = (
    "You are an outstanding mathematics teacher with deep insight and patient guidance. "
    "Stay confident, think carefully, and solve the problem well.\n\n"
)
IRRELEVANT_PREFIX = (
    "The weather is nice today. Let's work on a problem together.\n\n"
)

# Modes that never inject ground-truth privilege text.
NO_GT_MODES = {
    "same",
    "encourage",
    "irrelevant",
    "same_trans",
    "encourage_trans",
    "irrelevant_trans",
    "sample_irrelevant_trans",
}


class SelfDistillationDataCollator:
    """Build Qwen3 student/privileged-teacher prompts for an OPSD batch."""

    MODES = {
        "correct",
        "correct_simple",
        "pi",
        "instruction",
        "opsd",
        "same",
        "encourage",
        "irrelevant",
        "same_trans",
        "encourage_trans",
        "irrelevant_trans",
        "sample_irrelevant_trans",
    }

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 9216,
        max_prompt_length: int = 1024,
        privilege_mode: str = "correct",
        teacher_privilege_field: str = "solution",
        student_thinking: bool = False,
        teacher_thinking: bool = False,
        teacher_tokenizer: Any | None = None,
        **_: Any,
    ) -> None:
        if privilege_mode not in self.MODES:
            raise ValueError(f"privilege_mode must be one of {sorted(self.MODES)}")
        if teacher_privilege_field not in PRIVILEGE_FIELDS:
            raise ValueError(f"teacher_privilege_field must be one of {sorted(PRIVILEGE_FIELDS)}")
        self.tokenizer = tokenizer
        # Optional separate tokenizer for cross-model distillation (e.g. Instruct student + 4b teacher).
        self.teacher_tokenizer = teacher_tokenizer if teacher_tokenizer is not None else tokenizer
        self.max_length = int(max_length)
        self.max_prompt_length = int(max_prompt_length)
        self.privilege_mode = privilege_mode
        self.teacher_privilege_field = teacher_privilege_field
        self.student_thinking = bool(student_thinking)
        self.teacher_thinking = bool(teacher_thinking)
        self.reason_first = False
        self.tokenizer.padding_side = "right"
        self.teacher_tokenizer.padding_side = "right"

    def privilege_text(self, feature: dict[str, Any]) -> str:
        """Select teacher privilege content: full trajectory (`solution`) or short `answer`."""
        field = self.teacher_privilege_field
        if field == "none" or self.privilege_mode in NO_GT_MODES:
            return ""
        if field == "answer":
            for key in ("answer", "Answer"):
                if feature.get(key) is not None and str(feature[key]).strip():
                    return str(feature[key]).strip()
            # Backward compatible: answer-only parquet may store Answer in `solution`.
            return str(feature.get("solution", "")).strip()
        if feature.get("solution") is not None and str(feature["solution"]).strip():
            return str(feature["solution"]).strip()
        return ""

    @staticmethod
    def _standard_student_user(problem: str) -> str:
        """Official-style user content: problem + boxed instruction (no GT)."""
        return (
            f"Problem: {problem}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

    @staticmethod
    def _concise_answer_teacher_user(problem: str, answer: str) -> str:
        """Minimal privileged teacher: problem + answer + Qwen3 boxed instruction."""
        return (
            f"Problem: {problem}\n\n"
            f"Answer: {answer}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

    @staticmethod
    def _opsd_teacher_user(problem: str, privileged: str) -> str:
        """Official OPSD teacher scaffold. Empty `privileged` → no reference-solution block."""
        if str(privileged).strip():
            return (
                f"Problem: {problem}\n\n"
                "Here is a reference solution to this problem:\n"
                f"=== Reference Solution Begin ===\n{privileged}\n=== Reference Solution End ===\n"
                f"{OFFICIAL_TRANSITION_PROMPT}\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
        # No-GT: never emit reference-solution header/body.
        return SelfDistillationDataCollator._nogt_transition_teacher_user(problem)

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        """Ensure teacher prefix ends with a blank line before Problem:."""
        text = str(prefix or "").strip()
        if not text:
            return ""
        return text + "\n\n"

    @staticmethod
    def _nogt_transition_teacher_user(problem: str, prefix: str = "") -> str:
        """Problem (+ optional prefix) + no-GT transition; no reference-solution scaffold."""
        return (
            f"{prefix}"
            f"Problem: {problem}\n"
            f"{NO_GT_TRANSITION_PROMPT}\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

    def _sampled_irrelevant_prefix(self, feature: dict[str, Any]) -> str:
        """Prefer per-row `irrelevant_prefix` from preprocessed data; else fixed IRRELEVANT_PREFIX."""
        raw = feature.get("irrelevant_prefix")
        if raw is None or not str(raw).strip():
            return IRRELEVANT_PREFIX
        return self._normalize_prefix(str(raw))

    def format_prompts(self, feature: dict[str, Any]) -> tuple[str, str]:
        problem = str(feature["problem"]).strip()
        privileged = self.privilege_text(feature)

        if self.privilege_mode in NO_GT_MODES:
            # Student: plain problem prompt. Teacher: same / prefix / (+ transition, no GT).
            student_user = self._standard_student_user(problem)
            if self.privilege_mode == "same":
                teacher_user = student_user
            elif self.privilege_mode == "encourage":
                teacher_user = f"{ENCOURAGE_PREFIX}{student_user}"
            elif self.privilege_mode == "irrelevant":
                teacher_user = f"{IRRELEVANT_PREFIX}{student_user}"
            elif self.privilege_mode == "same_trans":
                teacher_user = self._nogt_transition_teacher_user(problem)
            elif self.privilege_mode == "encourage_trans":
                teacher_user = self._nogt_transition_teacher_user(problem, prefix=ENCOURAGE_PREFIX)
            elif self.privilege_mode == "irrelevant_trans":
                teacher_user = self._nogt_transition_teacher_user(problem, prefix=IRRELEVANT_PREFIX)
            else:  # sample_irrelevant_trans
                teacher_user = self._nogt_transition_teacher_user(
                    problem, prefix=self._sampled_irrelevant_prefix(feature)
                )
        elif self.privilege_mode == "opsd":
            # Official OPSD student / teacher templates (siyan-zhao/OPSD).
            # privilege_field=none → same no-GT transition template (no reference block).
            student_user = self._standard_student_user(problem)
            teacher_user = self._opsd_teacher_user(problem, privileged)
        elif self.privilege_mode == "instruction":
            student_instruction = (
                "Give a concise solution with only the essential reasoning, and put the final answer "
                "within \\boxed{}."
            )
            teacher_user = (
                f"Problem: {problem}\n\n"
                "Give a detailed, rigorous solution. Explain every important derivation, check the result, "
                "and put the final answer within \\boxed{}."
            )
            student_user = f"Problem: {problem}\n\n{student_instruction}"
        elif self.privilege_mode == "correct_simple":
            # Concise answer privilege: problem + answer + please reason step by step
            student_user = self._standard_student_user(problem)
            teacher_user = self._concise_answer_teacher_user(problem, privileged)
        else:
            student_instruction = (
                "Please reason step by step and put the final answer within \\boxed{}."
            )
            content = privileged if self.privilege_mode == "correct" else "π"
            label = "verified answer" if self.privilege_mode == "correct" else "privileged answer"
            teacher_user = (
                f"Problem: {problem}\n\n"
                f"Here is the {label}:\n"
                f"{content}\n\n"
                "After understanding the privileged information, solve the problem using your own reasoning. "
                "Please reason step by step and put the final answer within \\boxed{}."
            )
            student_user = f"Problem: {problem}\n\n{student_instruction}"

        student_prompt = self._apply_chat_template(
            self.tokenizer,
            [{"role": "user", "content": student_user}],
            enable_thinking=self.student_thinking,
        )
        teacher_prompt = self._apply_chat_template(
            self.teacher_tokenizer,
            [{"role": "user", "content": teacher_user}],
            enable_thinking=self.teacher_thinking,
        )
        return student_prompt, teacher_prompt

    @staticmethod
    def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], *, enable_thinking: bool) -> str:
        """apply_chat_template with enable_thinking when supported (Qwen3); Olmo ignores it."""
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)

    def prompt_lengths(self, feature: dict[str, Any]) -> tuple[int, int]:
        student, teacher = self.format_prompts(feature)
        return (
            len(self.tokenizer(student, add_special_tokens=False)["input_ids"]),
            len(self.teacher_tokenizer(teacher, add_special_tokens=False)["input_ids"]),
        )

    def fits(self, feature: dict[str, Any]) -> bool:
        student_len, teacher_len = self.prompt_lengths(feature)
        return student_len <= self.max_prompt_length and teacher_len <= self.max_prompt_length

    def _encode(
        self,
        prompts: list[str],
        *,
        tokenizer: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        tok = self.tokenizer if tokenizer is None else tokenizer
        no_pad = tok(prompts, padding=False, truncation=False, add_special_tokens=False)
        lengths = [len(ids) for ids in no_pad["input_ids"]]
        if max(lengths) > self.max_prompt_length:
            raise ValueError(
                f"prompt exceeds max_prompt_length={self.max_prompt_length}; "
                "filter the dataset with collator.fits before training"
            )
        encoded = tok(
            prompts,
            padding="longest",
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"], lengths

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        pairs = [self.format_prompts(feature) for feature in features]
        student_ids, student_mask, student_lengths = self._encode(
            [x[0] for x in pairs], tokenizer=self.tokenizer
        )
        teacher_ids, teacher_mask, teacher_lengths = self._encode(
            [x[1] for x in pairs], tokenizer=self.teacher_tokenizer
        )
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
