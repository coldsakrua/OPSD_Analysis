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
