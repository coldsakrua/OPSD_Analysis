# OPSD Student/Teacher Data Analysis

Analysis code for sections **2.1–2.5**: on-policy student rollouts scored against teacher prompts with KL/JSD, top-k KL (k=1,16), argmax preference, SNR, and loss-dominant token stats.

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
| `jsd_kl` | `generalized_jsd_loss` full vocab, T=1.1; **β=0.0** (matches `train_opsd.py`; forward KL). Note: folder `jsd005` / `0.05` is **`jsd_token_clip`**, not β |
| `topk_kl_k16` | Teacher top-16, renormalized, D_KL(p_T ‖ p_S) |
| `log_ratio_k1` | log π_S(x) − log π_T(x) for sampled token x |
| `advantage` | log π_T(x) − log π_S(x) (= −log_ratio_k1) |
| `snr` | \|advantage\| / (student_entropy + 1e-8), per token |
| `top1_agree_rate` | Fraction of positions where student_argmax == teacher_argmax |
| `top_disagree_*` | When argmax differs: preferred tokens / pairs (student vs teacher) |
| `frac_encourage` / `frac_discourage` | Fraction of tokens with advantage > 0 / < 0 |
| `top_encourage_*` / `top_discourage_*` | Tokens most often encouraged / discouraged by teacher |
| `topk_hit` | Sampled token ∈ TopK for student/teacher (K=4,8,16,32,64) |
| `teacher_hit_at_max_k` | Advantage split when sampled token ∈ teacher Top-max_k |
| `rank_within_topk_max` | Rank of sampled token in Top-max_k (capped if outside) |
| `mean_entropy_gap` | H(π_S) − H(π_T) — calibration / uncertainty mismatch |
| `mean_*_confidence_gap` | max log π − log π(x) — how far sampled token is from argmax |
| `frac_jsd_clipped` | Tokens with jsd_kl > jsd_token_clip (default 0.05, matches train) |
| `mean_jsd_kl_clipped` | Mean JSD after per-token clip |
| `top_loss_dominant_tokens` | Tokens with highest Σ divergence per token id |

Outputs:
- `summary.json` — global aggregates above
- `rollout_metrics.jsonl` — per-rollout means, frac_encourage, first/last-128 advantage, clip rate
- `token_metrics.jsonl` — full per-token arrays (disable with `--no-save-token-metrics`)

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
- Batch sizes auto-tune by **task + model size** on A800 80GB (override with `SCORE_BATCH` / `GEN_BATCH_HINT`):

| Task | Model tier | SCORE_BATCH | GEN_BATCH_HINT (SGLang) |
|------|------------|-------------|-------------------------|
| 2.1–2.4 | small (≤1.7B) | 8 | 64 |
| 2.1–2.4 | medium (4B) | 4 | 64 |
| 2.1–2.4 | large (7B) | 2 | 32 |
| 2.5 | small | 2 | 32 |
| 2.5 | medium | 1 | 16 |
| 2.5 | large | 1 | 8 |
