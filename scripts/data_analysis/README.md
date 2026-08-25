# OPSD Student/Teacher Data Analysis

Analysis code for sections **2.1–2.5**: on-policy student rollouts scored against teacher prompts with JSD KL, top-k KL (k=1,16), and loss-dominant token stats.

## Layout

```
scripts/data_analysis/
├── run_opsd_analysis.py      # Main Python entrypoint
├── _common_launch.sh         # Shared env + two-phase gen/score launcher
├── generate_shell_scripts.py # Regenerate section shell wrappers
├── common/                   # Shared Python modules
├── 2.1_combinations/         # Four st/tt combos per model
├── 2.2_teacher_prefix/       # sol / answer / irrelevant_other_sol
├── 2.3_entropy/              # he20, le20, he80, le80 buckets
├── 2.4_other_models/         # deepseek, falcon, mimo, qwen3-0.6b
└── 2.5_length_windows/       # Position buckets 0-128 … 4096-6144
```

Outputs: `scripts/data_analysis/outputs/<task>/<model>/<combo>_<jobid>/`

## Quick start

```bash
cd OPSD_Analysis

# Single job (Slurm)
sbatch scripts/data_analysis/2.1_combinations/analyze_st_tt_qwen3_1.7b.sh

# Local smoke (reduce prompts)
NUM_PROMPTS=32 N_ROLLOUTS=1 bash scripts/data_analysis/2.1_combinations/analyze_st_tt_qwen3_1.7b.sh
```

## Metrics

| Metric | Description |
|--------|-------------|
| `jsd_kl` | Generalized JSD (β=0.5), full vocabulary, T=1.1 |
| `topk_kl_k16` | Teacher top-16, renormalized, D_KL(p_T ‖ p_S) |
| `log_ratio_k1` | log π_S(x) − log π_T(x) for sampled token x |
| `top_loss_dominant_tokens` | Tokens with highest Σ JSD per token id |

## Regenerate shell scripts

```bash
python scripts/data_analysis/generate_shell_scripts.py
```

## Assumptions

- Models under `/gpfs/share/home/2501210611/labShare/2501210611/model/`
- Preprocessed parquet under `data/openthoughts/preprocessed/` (same paths as training scripts)
- Conda: `anchor` (Qwen), `qwen3_5` (Qwen3.5), `sglang` (Olmo/Mimo), `falcon` (Falcon-H1R; env matches train scripts)
- Default rollout: 2048 prompts × 2 rollouts, max_completion=1024 (sections 2.1–2.4)
- Section 2.5 length windows: max_prompt=1024, max_completion=6144 (matches length training)
