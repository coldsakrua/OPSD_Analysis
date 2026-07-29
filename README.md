# OPSD Qwen3-4B full-parameter training

This directory implements paper-style on-policy self-distillation for Qwen3-4B. The student samples its own trajectory with vLLM. A separate frozen copy of the initial checkpoint evaluates the same response tokens under a privileged prompt. The optimization target is full-vocabulary forward KL (`beta=0` in generalized JSD), with gradients only through the student.

## Experiment matrix

| Script | Teacher context | Qwen3 thinking |
|---|---|---|
| `scripts/train/opsd_nothink_4b.sh` | verified integer answer | student/teacher off |
| `scripts/train/opsd_think_4b.sh` | verified integer answer | student/teacher on |
| `scripts/train/opsd_student_nothink_teacher_think_4b.sh` | verified integer answer | **student off, teacher on** (paper-preferred) |
| `scripts/train/opsd_pi_nothink_4b.sh` | fixed wrong answer `π` | student/teacher off |
| `scripts/train/opsd_pi_think_4b.sh` | fixed wrong answer `π` | student/teacher on |
| `scripts/train/opsd_instruction_nothink_4b.sh` | detailed instruction, no answer | student/teacher off |
| `scripts/train/opsd_instruction_think_4b.sh` | detailed instruction, no answer | student/teacher on |

By default student and teacher share one `enable_thinking` switch. The asymmetric script sets `--no-student-thinking --teacher-thinking` to match the paper's preferred TM-off student / TM-on teacher pairing. In the instruction-shift variants, the student is instructed to be concise and the teacher to be detailed.

## Training configuration

- Model: `/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b`
- 2×A800, DeepSpeed ZeRO-3, full student parameter updates
- Frozen full-model teacher initialized from step 0
- Per-device batch 4, gradient accumulation 4, global batch 32 on 2 GPUs
- Prompt cap 1024, response cap 1024
- 100 optimizer steps; full checkpoints at 25, 50, 75, and 100
- AdamW, learning rate `5e-6`, cosine decay, 10% warmup, bf16
- vLLM colocate, TP=1 per rank, utilization `0.55`, sleep mode enabled
- Training attention: PyTorch SDPA; vLLM backend: XFormers; no FlashAttention execution
- WandB offline plus Trainer/Slurm logs
- No task reward, correctness signal, advantage, or training-time validation

Logged metrics include loss, student/teacher entropy, sampled-token log-probabilities, forward/reverse KL diagnostics, valid response tokens, generated length, rollout throughput, learning rate, gradient norm, and CUDA allocated/reserved/peak memory.

## Submit

Set the server dataset path and submit one experiment:

```bash
sbatch scripts/train/opsd_nothink_4b.sh
```

Submit from the repository root so Slurm writes job logs under `log/opsd_<jobname>.<jobid>.out`.

Default training data for `opsd_nothink_4b` is
`${BASE_DIR}/data/dapo/preprocessed/dapo-math-17k.opsd.correct.nothink.maxprompt1024.parquet`
(offline `{problem, solution}` + prompt-length filter). The paper-preferred asymmetric script
`opsd_student_nothink_teacher_think_4b.sh` uses
`dapo-math-17k.opsd.correct.snothink_tthink.maxprompt1024.parquet`. Rebuild with:

```bash
python scripts/data/preprocess_opsd_dapo.py --privilege-mode correct
# student TM-off + teacher TM-on:
python scripts/data/preprocess_opsd_dapo.py --privilege-mode correct --no-student-thinking --teacher-thinking
# field-only (no length filter):
python scripts/data/preprocess_opsd_dapo.py --skip-prompt-length-filter
```

Override with `DATASET_PATH` if needed.

### OpenThoughts (official OPSD prompts)

Full-parameter training on the local OpenThoughts dump (`data/openthoughts/`, same schema as
[`siyanzhao/Openthoughts_math_30k_opsd`](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd))
uses `privilege_mode=opsd` — the official student / reference-solution teacher templates from
[siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD).

```bash
# full trajectory privilege (official OPSD template)
python scripts/data/preprocess_opsd_openthoughts.py --teacher-privilege-field solution --no-student-thinking --teacher-thinking
sbatch scripts/train/opsd_student_nothink_teacher_think_4b_1e_6_openthoughts.sh

# answer-only privilege (same codepath; switch --teacher-privilege-field answer)
python scripts/data/preprocess_opsd_openthoughts.py --teacher-privilege-field answer --no-student-thinking --teacher-thinking
sbatch scripts/train/opsd_student_nothink_teacher_think_4b_1e_6_openthoughts_answer.sh
```

Both preprocessed parquets keep `problem` / `solution` / `answer`. Training selects privilege content with `--teacher-privilege-field {solution,answer}`.

Hyperparameters match `opsd_student_nothink_teacher_think_4b_1e_6.sh` (lr=`1e-6`, 100 steps, etc.).

Optional overrides are `BASE_DIR`, `DATASET_PATH`, `OUTPUT_ROOT`, `WANDB_DIR`, `HF_HOME`, and `MASTER_PORT`.

Before the first training job, verify the existing `anchor` environment without installing anything:

```bash
bash scripts/smoke_test_imports.sh
```

## Evaluation

The evaluator is adapted from CAST and directly loads a full checkpoint. It covers AIME24, AIME25, AIME26, HMMT25, and MATH500 with pass@1/4/8/16.

```bash
export CHECKPOINT_PATH=/path/to/opsd_nothink_4b/checkpoint-100
sbatch scripts/eval/eval_nothink.sh
```

Default eval data root is `${BASE_DIR}/data` (AIME24/25/26, HMMT25, MATH-500). Override with `EVAL_DATA_ROOT` if needed.

Use `scripts/eval/eval_think.sh` for a thinking checkpoint. No LoRA adapter argument is used.

## Layout

- `src/`: training Python (`train_opsd.py`, `opsd_trainer.py`, `opsd_config.py`, `data_collator.py`)
- `scripts/train/`: self-contained Slurm training jobs (no shared `*_common.sh`)
- `scripts/eval/`: self-contained Slurm evaluation jobs
- `eval/`: CAST-derived full-model math evaluation Python
- `configs/`: ZeRO-3 and Accelerate configuration
- `vendor/verl`: safely extracted CAST `verl.zip`
- `vendor/OPSD_official`: upstream OPSD reference snapshot
- `vendor/trl_v0.22.1`: server-version API reference used for static compatibility checks

## Olmo-3-7B-Instruct (SGLang rollout)

Olmo training stays on the same accelerate + DeepSpeed ZeRO-3 + full-parameter OPSD path as Qwen3-4B-Instruct, but uses **SGLang** for on-policy generation (vLLM does not run Olmo-3 well here).

- Conda env: `sglang` (torch 2.8 + sglang 0.5.4 + accelerate/trl/deepspeed/ray/verl)
- Smoke imports: `bash scripts/smoke_test_sglang_train_env.sh`
- Model: `/gpfs/share/home/2501210611/labShare/2501210611/model/olmo3-7b-it`
- Hyperparams match 4b-it OpenThoughts baseline (`lr=1e-6`, `jsd_clip=1e-6`, micro=4, gas=4, 100 steps) on **4 GPUs** (global batch 64)
- Train script: `scripts/train/olmo3_7b_instruct/openthoughts/opsd_student_nothink_teacher_nothink_1e_6_openthoughts.sh`

```bash
# preprocess (Olmo tokenizer length filter)
PYTHONPATH=src:vendor/verl python scripts/data/preprocess_opsd_openthoughts.py \
  --privilege-mode opsd --teacher-privilege-field solution \
  --no-student-thinking --no-teacher-thinking \
  --model-path /gpfs/share/home/2501210611/labShare/2501210611/model/olmo3-7b-it \
  --output data/openthoughts/preprocessed/openthoughts.opsd.solution.nothink.olmo7bit.maxprompt1024.parquet

# 2-step smoke
MAX_STEPS=2 sbatch scripts/train/olmo3_7b_instruct/openthoughts/opsd_student_nothink_teacher_nothink_1e_6_openthoughts.sh

# full 100-step run
sbatch scripts/train/olmo3_7b_instruct/openthoughts/opsd_student_nothink_teacher_nothink_1e_6_openthoughts.sh
```

If OOM, keep lr/jsd_clip fixed and lower `PER_DEVICE_BATCH_SIZE` or `SGLANG_MEM_FRACTION_STATIC`.
