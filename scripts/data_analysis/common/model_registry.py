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
# 2.2: short solution / answer / distractors / long solution (teacher cap 12*1024, cf. openmath train).
TEACHER_PREFIXES = ("sol", "answer", "irrelevant_other_sol", "sol_long")
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
# Match OT cotlen-easy/hard training (scripts/train/*/jsd005/*/…_cotlen_{easy,hard}.sh).
COTLEN_MAX_PROMPT = 2048
# Match openmath OPSD train (…_openmath.sh): teacher may embed full reference solution.
LONG_TEACHER_MAX_PROMPT = 12 * 1024  # 12288
COTLEN_BANDS = ("easy", "hard")

# Match project eval max_new_tokens (scripts/eval/{1.7b,4b,4b_instruct,olmo*}).
EVAL_MAX_COMPLETION: dict[str, int] = {
    "qwen3_1.7b": 38912,
    "qwen3_4b": 38912,
    "qwen3_4b_instruct": 32768,
    "qwen3_8b": 38912,
    "olmo3_7b_think": 32768,
    "olmo3_7b_instruct": 32768,
}

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

# Train scripts named st_tt but chat templates ignore enable_thinking (falcon/mimo).
MODEL_COMBO_THINK_OVERRIDE: dict[str, dict[str, tuple[bool, bool]]] = {
    "falcon_h1r_7b": {"st_tt": (False, False)},
    "mimo_7b_rl": {"st_tt": (False, False)},
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
    ),
    "qwen3_06b": ModelConfig(
        "qwen3_06b",
        f"{MODEL_ROOT}/qwen3-0.6b",
        "anchor",
        "vllm",
        ".qwen3_06b",
    ),
    "qwen3_8b": ModelConfig(
        "qwen3_8b",
        f"{MODEL_ROOT}/qwen3-8b",
        "anchor",
        "vllm",
        ".qwen3_8b",
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
        "qwen3_4b": ("st_tt", "snt_tnt"),
        "qwen3_4b_instruct": ("snt_tnt",),
        "qwen3_4b_thinking": ("st_tt",),
        "qwen3_06b": ("st_tt",),
        "deepseek_r1_1.5b": ("st_tt",),
        "mimo_7b_rl": ("st_tt",),
        "falcon_h1r_7b": ("st_tt",),
    },
    "2.3": {
        "qwen3_1.7b": ("st_tt", "snt_tnt"),
        "qwen3_4b": ("st_tt", "snt_tnt"),
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
    # OT gold-CoT length bands (D0-4 easy / D7-9 hard); matches train …_cotlen_{easy,hard}.sh
    "2.6": {
        "qwen3_1.7b": ("st_tt",),
        "qwen3_4b": ("st_tt",),
        "qwen3_4b_instruct": ("snt_tnt",),
        "olmo3_7b_think": ("st_tt",),
        "olmo3_7b_instruct": ("snt_tnt",),
        # Train scripts exist; preprocessed parquet may be missing until preprocess.
        "qwen3_8b": ("st_tt",),
    },
}

# Middle filename segment for cotlen preprocessed parquets (maxprompt2048).
# Distinct from maxprompt1024 tags used by sections 2.1–2.5.
COTLEN_DATASET_TAG: dict[tuple[str, str], str] = {
    ("qwen3_1.7b", "st_tt"): "sthink_tthink.qwen3_1.7b",
    ("qwen3_4b", "st_tt"): "sthink_tthink.qwen3_4b",
    ("qwen3_4b_instruct", "snt_tnt"): "nothink.qwen3_4b_instruct",
    ("olmo3_7b_think", "st_tt"): "sthink_tthink.olmo3_7b_think",
    ("olmo3_7b_instruct", "snt_tnt"): "nothink.olmo3_7b_instruct",
    ("qwen3_8b", "st_tt"): "sthink_tthink.qwen3_8b",
}


def get_model_config(model_key: str) -> ModelConfig:
    if model_key not in MODELS:
        raise KeyError(f"unknown model {model_key!r}; known: {sorted(MODELS)}")
    return MODELS[model_key]


def combo_think(model_key: str, combo: str) -> tuple[bool, bool]:
    """Student/teacher thinking flags; match train THINK_ARGS per model."""
    overrides = MODEL_COMBO_THINK_OVERRIDE.get(model_key, {})
    if combo in overrides:
        return overrides[combo]
    return COMBO_THINK[combo]


def _dataset_tag_part(model: ModelConfig, combo: str) -> str:
    """Middle segment of opsd/solution parquet filename (before .maxprompt1024)."""
    suffix = model.dataset_suffix
    if suffix.startswith(".nothink."):
        # e.g. .nothink.instruct → nothink.instruct (combo tag omitted; train uses snt_tnt parquet)
        return "nothink" + suffix[len(".nothink") :]
    if suffix:
        return f"{COMBO_DATASET_TAG[combo]}{suffix}"
    return COMBO_DATASET_TAG[combo]


def dataset_path(
    model_key: str,
    combo: str,
    base_dir: Path | None = None,
    *,
    max_prompt: int = DEFAULT_MAX_PROMPT,
) -> Path:
    model = get_model_config(model_key)
    tag_part = _dataset_tag_part(model, combo)
    root = base_dir or DATA_ROOT
    name = f"openthoughts.opsd.solution.{tag_part}.maxprompt{int(max_prompt)}.parquet"
    return Path(root) / name


def teacher_prefix_max_prompt(prefix: str, *, default_max_prompt: int = DEFAULT_MAX_PROMPT) -> int:
    """Per-prefix teacher prompt cap for section 2.2."""
    if prefix == "sol_long":
        return LONG_TEACHER_MAX_PROMPT
    if prefix in TEACHER_PREFIXES:
        return int(default_max_prompt)
    raise ValueError(f"unknown teacher prefix {prefix!r}")


def cotlen_dataset_path(
    model_key: str,
    combo: str,
    band: str,
    *,
    max_prompt: int = COTLEN_MAX_PROMPT,
    base_dir: Path | None = None,
) -> Path:
    """OT cotlen-easy/hard parquet used by train …_cotlen_{easy,hard}.sh.

    Always points at the train-preprocessed maxprompt2048 files (problem pool).
    Analysis may still use a shorter --max-prompt-length (e.g. 1024 like 2.2).
    """
    if band not in COTLEN_BANDS:
        raise ValueError(f"unknown cotlen band {band!r}; expected {COTLEN_BANDS}")
    tag = COTLEN_DATASET_TAG.get((model_key, combo))
    if tag is None:
        raise KeyError(
            f"no cotlen dataset tag for model={model_key!r} combo={combo!r}; "
            f"known={sorted(COTLEN_DATASET_TAG)}"
        )
    root = base_dir or DATA_ROOT
    # Train scripts use maxprompt2048; keep that filename regardless of analysis prompt cap.
    _ = max_prompt
    name = f"openthoughts.cotlen_{band}.opsd.solution.{tag}.maxprompt{COTLEN_MAX_PROMPT}.parquet"
    return Path(root) / name


def teacher_prefix_dataset(
    model_key: str,
    combo: str,
    prefix: str,
    base_dir: Path | None = None,
    *,
    default_max_prompt: int = DEFAULT_MAX_PROMPT,
) -> Path:
    """Dataset parquet for teacher-prefix variants (2.2)."""
    model = get_model_config(model_key)
    tag_part = _dataset_tag_part(model, combo)
    root = base_dir or DATA_ROOT
    short = int(default_max_prompt)
    if prefix == "sol":
        return dataset_path(model_key, combo, base_dir=root, max_prompt=short)
    if prefix == "sol_long":
        return dataset_path(
            model_key, combo, base_dir=root, max_prompt=LONG_TEACHER_MAX_PROMPT
        )
    if prefix == "answer":
        return Path(root) / f"openthoughts.correct.answer.{tag_part}.maxprompt{short}.parquet"
    if prefix == "irrelevant_other_sol":
        return (
            Path(root)
            / f"openthoughts.irrelevant_other_sol.{tag_part}.maxprompt{short}.parquet"
        )
    raise ValueError(f"unknown teacher prefix {prefix!r}")


def privilege_for_prefix(prefix: str) -> tuple[str, str]:
    """Return (privilege_mode, teacher_privilege_field) for collator."""
    if prefix in ("sol", "sol_long"):
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


def task_default_max_completion(task: str, model_key: str | None = None) -> int:
    """Default rollout cap per analysis section."""
    if task == "length_windows":
        return LENGTH_WINDOWS_MAX_COMPLETION
    if task == "cotlen":
        if not model_key:
            raise ValueError("model_key required for cotlen max_completion")
        if model_key not in EVAL_MAX_COMPLETION:
            raise KeyError(f"no eval max_completion for {model_key!r}; extend EVAL_MAX_COMPLETION")
        return EVAL_MAX_COMPLETION[model_key]
    return DEFAULT_MAX_COMPLETION


def cotlen_max_completion(model_key: str) -> int:
    return task_default_max_completion("cotlen", model_key)


def task_default_max_prompt(task: str) -> int:
    """Default prompt cap. Cotlen uses 2048 (train); other sections use 1024."""
    if task == "cotlen":
        return COTLEN_MAX_PROMPT
    return DEFAULT_MAX_PROMPT


MODEL_SIZE_TIER: dict[str, str] = {
    "qwen3_06b": "small",
    "deepseek_r1_1.5b": "small",
    "qwen3_1.7b": "small",
    "qwen3_4b": "medium",
    "qwen3_4b_instruct": "medium",
    "qwen3_4b_thinking": "medium",
    "qwen3.5_4b": "medium",
    "qwen3_8b": "large",
    "olmo3_7b_instruct": "large",
    "olmo3_7b_think": "large",
    "falcon_h1r_7b": "large",
    "mimo_7b_rl": "large",
}

# HF score microbatch: short tasks (max_completion=1024) vs long (2.5 / cotlen-eval lengths).
# Peak VRAM scales ~B * L * vocab during forward + metric temps on GPU.
# Attention is ~B * S^2, so long teacher prompts need smaller B than short prefixes.
_SCORE_BATCH_SHORT = {"small": 8, "medium": 4, "large": 2}
_SCORE_BATCH_LONG = {"small": 2, "medium": 1, "large": 1}
_SCORE_BATCH_EVAL_LONG = {"small": 1, "medium": 1, "large": 1}
_GEN_BATCH_SHORT = {"small": 64, "medium": 64, "large": 32}
_GEN_BATCH_LONG = {"small": 32, "medium": 16, "large": 8}
_GEN_BATCH_EVAL_LONG = {"small": 8, "medium": 4, "large": 2}

# Approx parameter count (B) for A800-80GB sol_long batch sizing.
_MODEL_PARAMS_B: dict[str, float] = {
    "qwen3_06b": 0.6,
    "deepseek_r1_1.5b": 1.5,
    "qwen3_1.7b": 1.7,
    "qwen3_4b": 4.0,
    "qwen3_4b_instruct": 4.0,
    "qwen3_4b_thinking": 4.0,
    "qwen3.5_4b": 4.0,
    "qwen3_8b": 8.0,
    "olmo3_7b_instruct": 7.0,
    "olmo3_7b_think": 7.0,
    "falcon_h1r_7b": 7.0,
    "mimo_7b_rl": 7.0,
}


def model_params_b(model_key: str) -> float:
    if model_key not in _MODEL_PARAMS_B:
        raise KeyError(f"unknown param count for {model_key!r}; extend _MODEL_PARAMS_B")
    return _MODEL_PARAMS_B[model_key]


def score_batch_for_sol_long(model_key: str) -> int:
    """HF score microbatch for sol_long (teacher≤12288 + completion≤1024) on A800 80GB.

    Peak is dominated by one full forward on S≈13k (attention ~S²) plus transient [B,S,V]
    logits. Calibrated from the short-prefix table (S≈2k → 8/4/2) with a small-model
    boost because weight memory is far below 80GB:

      params_B   short@2k   linear*(2k/13k)   chosen sol_long
      0.6B         8           ~1.2              8
      1.5–1.7B     8           ~1.2              4
      4B           4           ~0.6              2
      7–8B         2           ~0.3              1
    """
    params = model_params_b(model_key)
    if params <= 0.8:
        return 8
    if params <= 2.0:
        return 4
    if params <= 5.0:
        return 2
    return 1


def score_batch_for_teacher_prefix(model_key: str, prefix: str) -> int:
    """Per-prefix score batch for section 2.2."""
    if prefix == "sol_long":
        return score_batch_for_sol_long(model_key)
    return task_default_score_batch("teacher_prefix", model_key)


def model_size_tier(model_key: str) -> str:
    tier = MODEL_SIZE_TIER.get(model_key)
    if tier is None:
        raise KeyError(f"unknown model tier for {model_key!r}; extend MODEL_SIZE_TIER")
    return tier


def task_default_score_batch(task: str, model_key: str) -> int:
    """HF scoring microbatch size tuned for A800 80GB."""
    tier = model_size_tier(model_key)
    if task == "cotlen":
        table = _SCORE_BATCH_EVAL_LONG
    elif task == "length_windows":
        table = _SCORE_BATCH_LONG
    elif task == "teacher_prefix":
        # Short prefixes only; sol_long uses score_batch_for_sol_long().
        table = _SCORE_BATCH_SHORT
    else:
        table = _SCORE_BATCH_SHORT
    return table[tier]


def task_default_gen_batch_hint(task: str, model_key: str) -> int:
    """SGLang prompt chunk size (vLLM ignores this; uses continuous batching)."""
    tier = model_size_tier(model_key)
    if task == "cotlen":
        table = _GEN_BATCH_EVAL_LONG
    elif task == "length_windows":
        table = _GEN_BATCH_LONG
    else:
        table = _GEN_BATCH_SHORT
    return table[tier]


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
