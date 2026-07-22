# veRL RLSD/OPSD/RLRT/CAST launchers

This folder contains the veRL Python launch code for CAST training experiments.

## Layout

| Path | Purpose |
| --- | --- |
| `model/` | Local model weights (`qwen3-4b`, `deepseek-math-7b-rl`) |
| `configs/` | YAML experiment configs (`train/`, `eval/`, `models.yaml`) |
| `scripts/sbatch/` | 自包含 sbatch 脚本（`#SBATCH` + 环境 + `run_train/eval.py`，参数在 `configs/`） |
| `scripts/local/` | 同上，无 sbatch，本地直跑 |
| `scripts/lib/` | `run_train.py`, `run_eval.py`, `load_config.py`（Python，被 sh 直接调用） |
| `train_scripts/` | Compatibility forwarders → `scripts/sbatch/train/` |
| `eval_scripts/` | Compatibility forwarders → `scripts/sbatch/eval/` |

## Files

- `verl_rlsd/reward.py`: math reward adapter for veRL custom rewards.
- `verl_rlsd/advantage.py`: custom advantage estimators:
  - `rlsd_grpo`: paper RLSD/OPSD token-gap shaping on GRPO advantages.
  - `rlsd_strict_split_flip`: strict split fallback plus correct-path sign flip.
  - `rlsd_strict_split_flip_wrong_boost`: strict split plus wrong-path positive flip.
  - `rlrt`: RLRT reversed teacher weighting on correct rollouts.
  - `opd_zero`: zero policy-gradient reward for distillation-only OPSD.
- `verl_rlsd/teacher_agent.py`: custom veRL agent loop manager.
- `verl_rlsd/teacher_ema.py`: EMA teacher LoRA sync.

## Qwen 4B (`configs/train/qwen/`)

| Config / script | Main behavior |
| --- | --- |
| `grpo_4b` | Strict GRPO baseline, no teacher shaping. |
| `rlsd_4b` | Canonical RLSD paper token shaping, lambda 0.5, decay 50. |
| `rlrt_4b` | RLRT reversed teacher weighting. |
| `opsd_4b` | Official OPSD-style distillation-only run, n=1. |
| `grpo_opds_4b` | Pure OPSD-style token-gap shaping with reward. |
| `cast_teacher50_4b_full_nogap005` | CAST with internal teacher snapshot (lambda 0.5, decay 300). |

Default data: `data/dapo/dapo-math-17k.parquet`
Default model: `model/qwen3-4b`

## DeepSeek-Math-7B-RL (`configs/train/deepseek/`)

| Config | Main behavior |
| --- | --- |
| `grpo_deepseek_math_7b_rl_strict` | GRPO baseline on GSM8K. |
| `rlsd_deepseek_math_7b_rl_paper` | RLSD paper objective. |
| `rlrt_deepseek_math_7b_rl` | RLRT. |
| `opsd_deepseek_math_7b_rl` | OPSD distillation-only. |
| `rlsd_..._flip_wrong_boost_...` | CAST flip_wrong_boost, lambda 1.0, no decay. |
| `rlsd_..._strict_split_nodecay_...` | Strict split flip without wrong_boost. |
| `rlsd_..._phase1_300` | CAST 300 steps. |
| `grpo_deepseek_math_7b_rl_resume_from_300` | GRPO resume to 1200 steps. |

Default data: `data/gsm8k/main/train-00000-of-00001.parquet`
Default model: `model/deepseek-math-7b-rl`
DeepSeek configs set `RELAXED_ANSWER_EXTRACTION=true` and `STRIP_DAPO_PROMPT_BOILERPLATE=false`.

### Two-phase _300 workflow

```bash
cd /gpfs/share/home/2501210611/CAST

# Phase 1: CAST 300 steps (sbatch cluster)
sbatch scripts/sbatch/train/deepseek/rlsd_deepseek_math_7b_rl_strict_split_flip_wrong_boost_nodecay_no_teacher_ref_phase1_300.sh

# Phase 2: GRPO resume from phase1 job directory
OUTPUT_DIR=/gpfs/share/home/2501210611/CAST/verl_outputs/rlsd_deepseek_math_7b_rl_strict_split_flip_wrong_boost_nodecay_no_teacher_ref_300/job_XXX \
sbatch scripts/sbatch/train/deepseek/grpo_deepseek_math_7b_rl_resume_from_300.sh
```

## Submit on the cluster (sbatch)

```bash
sbatch scripts/sbatch/train/qwen/rlsd_4b.sh
sbatch scripts/sbatch/train/deepseek/rlsd_deepseek_math_7b_rl_paper.sh
```

Legacy paths still work (`train_scripts/qwen/rlsd_4b.sh` forwards to sbatch).

## Run on a normal server (no sbatch)

```bash
bash scripts/local/train/qwen/rlsd_4b.sh
bash scripts/local/eval/qwen/eval_math500_think_4b.sh
```

## Overrides

Environment variables override YAML defaults (same names as before):

```bash
BASE_DIR=/gpfs/share/home/2501210611/CAST \
MODEL_PATH=/gpfs/share/home/2501210611/CAST/model/qwen3-4b \
DATASET_PATH=/gpfs/share/home/2501210611/CAST/data/dapo/dapo-math-17k.parquet \
MAX_STEPS=300 \
sbatch scripts/sbatch/train/qwen/rlsd_4b.sh
```

Edit experiment defaults in `configs/train/.../*.yaml` or `configs/eval/.../*.yaml`.

Append raw veRL/Hydra overrides with `VERL_EXTRA_ARGS`.

## Data (`data/`)

| Path | Used by |
| --- | --- |
| `dapo/dapo-math-17k.parquet` | Qwen training |
| `gsm8k/main/train-*.parquet` | DeepSeek training |
| `gsm8k/main/test-*.parquet` | DeepSeek gsm8k eval |
| `AIME24/test.parquet` | aime24 eval |
| `AIME25/test.parquet` | aime25 eval |
| `AIME26/test.parquet` | aime26 eval |
| `HMMT25/test.parquet` | hmmt25 eval |
| `MATH-500/test.parquet` | math500 eval |

## Eval

### Qwen 4B (`configs/eval/qwen/`)

8 configs: 4 datasets × 2 modes (think, nothink), 32k context.

```bash
CHECKPOINT_DIR=/gpfs/share/home/2501210611/CAST/verl_outputs/cast_teacher50_4b_full_nogap005/job_XXX/global_step_300/actor/lora_adapter \
sbatch scripts/sbatch/eval/qwen/eval_aime24_think_4b.sh
```

### DeepSeek-Math-7B-RL (`configs/eval/deepseek/`)

4 configs with 4k fill-context eval:

| Config | Datasets |
| --- | --- |
| `eval_4k_aime24_aime26_deepseek_math_7b_rl` | aime24, aime26 |
| `eval_4k_aime25_hmmt25_deepseek_math_7b_rl` | aime25, hmmt25 |
| `eval_4k_math500_deepseek_math_7b_rl` | math500 |
| `eval_4k_gsm8k_deepseek_math_7b_rl` | gsm8k |

```bash
CHECKPOINT_DIR=/gpfs/share/home/2501210611/CAST/verl_outputs/rlsd_deepseek_math_7b_rl_strict_split_flip_wrong_boost_nodecay_no_teacher_ref/job_XXX/checkpoint-300 \
sbatch scripts/sbatch/eval/deepseek/eval_4k_aime24_aime26_deepseek_math_7b_rl.sh
```
