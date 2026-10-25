"""APPS code-generation prompt template for GRPO / eval.

Aligned with ``eval_code/lcb_runner/prompts/code_generation.py`` formatting
messages and the LCB verl preprocess wrapper in ``preprocess_lcb_v6.py``.
"""

from __future__ import annotations

FORMATTING_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters."
)

FORMATTING_STDIN = (
    "Read the inputs from stdin, solve the problem, and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters "
    "as follows. Ensure that when the python program runs, it reads the inputs, runs "
    "the algorithm and writes output to STDOUT."
)

APPS_CODE_PROMPT = (
    "You are a coding expert. You will be given a coding problem, and you need to write a "
    "correct Python program that matches the specification and passes all tests. "
    "You may start by outlining your thought process. In the end, please provide the "
    "complete code in a ```python``` code block.\n\n"
    "### Question:\n{question}\n\n"
    "### Format: {format_msg}\n"
    "```python\n{code_stub}\n```\n\n"
    "### Answer: (use the provided format with backticks)\n"
)


def build_apps_user_prompt(
    question: str,
    *,
    starter_code: str = "",
    fn_name: str | None = None,
) -> str:
    """Build the user-turn content for one APPS problem."""
    q = (question or "").strip()
    starter = (starter_code or "").strip()
    if starter:
        format_msg = FORMATTING_WITH_STARTER
        code_stub = starter
    elif fn_name:
        format_msg = (
            f"Implement a solution that exposes `{fn_name}` "
            f"(LeetCode-style `class Solution` is fine). "
            "Enclose your code within delimiters."
        )
        code_stub = f"class Solution:\n    def {fn_name}(self, *args, **kwargs):\n        ..."
    else:
        format_msg = FORMATTING_STDIN
        code_stub = "# YOUR CODE HERE"
    return APPS_CODE_PROMPT.format(
        question=q,
        format_msg=format_msg,
        code_stub=code_stub,
    )
