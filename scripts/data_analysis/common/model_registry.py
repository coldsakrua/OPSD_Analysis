"""Model paths, conda envs, generation backends, and dataset naming."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data" / "openthoughts" / "preprocessed"
MODEL_ROOT = Path("/gpfs/share/home/2501210611/labShare/2501210611/model")

COMBOS = ("st_tt", "snt_tnt", "st_tnt", "snt_tt")
ENTROPY_BUCKETS = ("he20", "le20", "he80", "le80")
TEACHER_PREFIXES = ("sol", "answer", "irrelevant_other_sol")
LENGTH_WINDOWS = (
    (0, 128),
    (128, 256),
    (256, 512),
    (512, 1024),
    (1024, 2048),
    (2048, 4096),
    (4096, 6144),
)

# Match length training (scripts/train/*/jsd005/length/): max_prompt=1024, max_completion=6144.
DEFAULT_MAX_PROMPT = 1024
DEFAULT_MAX_COMPLETION = 1024
LENGTH_WINDOWS_MAX_COMPLETION = 6144

COMBO_THINK = {
    "st_tt": (True, True),
    "snt_tnt": (False, False),
    "st_tnt": (True, False),
    "snt_tt": (False, True),
}

COMBO_DATASET_TAG = {
    "st_tt": "sthink_tthink",
    "snt_tnt": "nothink",
    "st_tnt": "sthink_tnothink",
    "snt_tt": "snothink_tthink",
}


@dataclass(frozen=True)
class ModelConfig:
    key: str
    model_path: str
    conda_env: str
    backend: str  # vllm | sglang
    dataset_suffix: str  # e.g. ".qwen35_4b" or ""
    reasoning_parser: str = ""
    conda_extras: dict[str, str] | None = None


MODELS: dict[str, ModelConfig] = {
    "qwen3_1.7b": ModelConfig(
        "qwen3_1.7b",
        f"{MODEL_ROOT}/qwen3-1.7b",
        "anchor",
        "vllm",
        "",
    ),
    "qwen3_4b": ModelConfig(
        "qwen3_4b",
        f"{MODEL_ROOT}/qwen3-4b",
        "anchor",
        "vllm",
        "",
    ),
    "qwen3_4b_instruct": ModelConfig(
        "qwen3_4b_instruct",
        f"{MODEL_ROOT}/qwen3-4b-instruct",
        "anchor",
        "vllm",
        ".nothink.instruct",
    ),
    "qwen3_4b_thinking": ModelConfig(
        "qwen3_4b_thinking",
        f"{MODEL_ROOT}/qwen3-4b-thinking",
        "anchor",
        "vllm",
        ".qwen3_4b_thinking",
    ),
    "qwen3.5_4b": ModelConfig(
        "qwen3.5_4b",
        f"{MODEL_ROOT}/qwen35_4b",
        "qwen3_5",
        "sglang",
        ".qwen35_4b",
        reasoning_parser="qwen3",
    ),
    "qwen3_06b": ModelConfig(
        "qwen3_06b",
        f"{MODEL_ROOT}/qwen3-0.6b",
        "anchor",
        "vllm",
        ".qwen3_06b",
    ),
    "olmo3_7b_instruct": ModelConfig(
        "olmo3_7b_instruct",
        f"{MODEL_ROOT}/olmo3-7b-it",
        "sglang",
        "sglang",
        ".nothink.olmo7bit",
    ),
    "olmo3_7b_think": ModelConfig(
        "olmo3_7b_think",
        f"{MODEL_ROOT}/olmo-3-7b-think",
        "sglang",
        "sglang",
        ".olmo7bthink",
    ),
    "deepseek_r1_1.5b": ModelConfig(
        "deepseek_r1_1.5b",
        f"{MODEL_ROOT}/deepseek-r1-distill-qwen-1.5b",
        "anchor",
        "vllm",
        ".deepseek_r1_1.5b",
    ),
    "falcon_h1r_7b": ModelConfig(
        "falcon_h1r_7b",
        f"{MODEL_ROOT}/falcon-h1r-7b",
        "falcon",
        "sglang",
        ".nothink.falconh1r7b",
        reasoning_parser="deepseek-r1",
    ),
    "mimo_7b_rl": ModelConfig(
        "mimo_7b_rl",
        f"{MODEL_ROOT}/mimo-7b-rl",
        "sglang",
        "sglang",
        ".nothink.mimo7brl",
    ),
}

# Section → model → allowed combos
SECTION_MODEL_COMBOS: dict[str, dict[str, tuple[str, ...]]] = {
    "2.1": {
        "qwen3_1.7b": COMBOS,
        "qwen3.5_4b": COMBOS,
        "olmo3_7b_instruct": ("snt_tnt",),
        "qwen3_4b_instruct": ("snt_tnt",),
        "qwen3_4b_thinking": ("st_tt",),
        "olmo3_7b_think": ("st_tt",),
    },
    "2.2": {
        "qwen3_1.7b": ("st_tt", "snt_tnt"),
        "olmo3_7b_instruct": ("snt_tnt",),
        "olmo3_7b_think": ("st_tt",),
        "qwen3_4b_instruct": ("snt_tnt",),
    },
    "2.3": {
        "qwen3_1.7b": ("st_tt",),
        "qwen3_4b": ("st_tt",),
        "qwen3_4b_instruct": ("snt_tnt",),
        "qwen3_4b_thinking": ("st_tt",),
        "olmo3_7b_instruct": ("snt_tnt",),
        "olmo3_7b_think": ("st_tt",),
    },
    "2.4": {
        "deepseek_r1_1.5b": ("st_tt",),
        "falcon_h1r_7b": ("st_tt",),
        "mimo_7b_rl": ("st_tt",),
        "qwen3_06b": ("st_tt",),
    },
    "2.5": {
        "qwen3_1.7b": ("st_tt",),
        "qwen3_4b": ("st_tt",),
        "qwen3_4b_instruct": ("snt_tnt",),
        "olmo3_7b_think": ("st_tt",),
        "olmo3_7b_instruct": ("snt_tnt",),
    },
}


def get_model_config(model_key: str) -> ModelConfig:
    if model_key not in MODELS:
        raise KeyError(f"unknown model {model_key!r}; known: {sorted(MODELS)}")
    return MODELS[model_key]


def _suffix_for_combo(model: ModelConfig, combo: str) -> str:
    """Return dataset filename suffix for opsd/solution parquet."""
    if model.dataset_suffix:
        return model.dataset_suffix
    tag = COMBO_DATASET_TAG[combo]
    if tag == "nothink":
        return ".nothink"
    return ""


def dataset_path(model_key: str, combo: str, base_dir: Path | None = None) -> Path:
    model = get_model_config(model_key)
    tag = COMBO_DATASET_TAG[combo]
    suffix = _suffix_for_combo(model, combo)
    root = base_dir or DATA_ROOT
    name = f"openthoughts.opsd.solution.{tag}{suffix}.maxprompt1024.parquet"
    return Path(root) / name


def teacher_prefix_dataset(
    model_key: str,
    combo: str,
    prefix: str,
    base_dir: Path | None = None,
) -> Path:
    """Dataset parquet for teacher-prefix variants (2.2)."""
    model = get_model_config(model_key)
    tag = COMBO_DATASET_TAG[combo]
    root = base_dir or DATA_ROOT
    if prefix == "sol":
        return dataset_path(model_key, combo, base_dir=root)
    if prefix == "answer":
        return Path(root) / f"openthoughts.correct.answer.{tag}.maxprompt1024.parquet"
    if prefix == "irrelevant_other_sol":
        return Path(root) / f"openthoughts.irrelevant_other_sol.{tag}.maxprompt1024.parquet"
    raise ValueError(f"unknown teacher prefix {prefix!r}")


def privilege_for_prefix(prefix: str) -> tuple[str, str]:
    """Return (privilege_mode, teacher_privilege_field) for collator."""
    if prefix == "sol":
        return "opsd", "solution"
    if prefix == "answer":
        return "correct", "answer"
    if prefix == "irrelevant_other_sol":
        return "irrelevant_other_sol", "none"
    raise ValueError(prefix)


def entropy_ratio(bucket: str) -> tuple[str, float]:
    """Return ('high'|'low', ratio) for entropy bucket name."""
    mapping = {
        "he20": ("high", 0.2),
        "he80": ("high", 0.8),
        "le20": ("low", 0.2),
        "le80": ("low", 0.8),
    }
    if bucket not in mapping:
        raise ValueError(f"unknown entropy bucket {bucket!r}")
    return mapping[bucket]


def task_default_max_completion(task: str) -> int:
    """Default rollout cap per analysis section."""
    if task == "length_windows":
        return LENGTH_WINDOWS_MAX_COMPLETION
    return DEFAULT_MAX_COMPLETION


def model_launch_overrides(model_key: str) -> dict[str, Any]:
    """Extra CLI flags for generation backend per model."""
    m = get_model_config(model_key)
    out: dict[str, Any] = {"backend": m.backend}
    if m.reasoning_parser:
        out["reasoning_parser"] = m.reasoning_parser
    if m.backend == "sglang":
        out["attention_backend"] = "triton"
        out["sampling_backend"] = "pytorch"
        out["mem_fraction_static"] = 0.80
    # Match scripts/train/falcon_h1r_7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts.sh
    if model_key == "falcon_h1r_7b":
        out["mem_fraction_static"] = 0.40
        out["disable_piecewise_cuda_graph"] = True
    return out
