from __future__ import annotations

from dataclasses import dataclass, field

from trl import SFTConfig


@dataclass
class OPSDConfig(SFTConfig):
    """GOLD-compatible arguments implemented against server TRL 0.22.1."""

    temperature: float = 1.1
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    lmbda: float = 1.0
    beta: float = 0.0
    max_completion_length: int = 8192
    student_model_revision: str | None = None
    disable_dropout: bool = True
    seq_kd: bool = False
    steps_per_generation: int | None = None
    use_transformers_paged: bool = False

    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_server_host: str = "0.0.0.0"
    vllm_server_port: int = 8001
    vllm_server_timeout: float = 240.0
    vllm_gpu_memory_utilization: float = 0.45
    vllm_tensor_parallel_size: int = 1
    vllm_guided_decoding_regex: str | None = None
    vllm_sync_frequency: int = 1
    vllm_enable_sleep_mode: bool = True

    log_completions: bool = False
    log_completions_steps: int = 25
    num_completions_to_print: int = 2
    wandb_entity: str | None = None
    wandb_project: str = "OPSD"
    wandb_run_group: str | None = None
    wandb_log_unique_prompts: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if self.max_length is not None and self.max_completion_length >= self.max_length:
            raise ValueError("max_length must exceed max_completion_length")
        if self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
        super().__post_init__()
