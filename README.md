# OPSD Qwen3-4B full-parameter training

This directory implements paper-style on-policy self-distillation for Qwen3-4B. The student samples its own trajectory with vLLM. A separate frozen copy of the initial checkpoint evaluates the same response tokens under a privileged prompt. The optimization target is full-vocabulary forward KL (`beta=0` in generalized JSD), with gradients only through the student.

## Experiment matrix

| Script | Teacher context | Qwen3 thinking |
|---|---|---|
| `scripts/opsd_correct_nothink.sh` | verified integer answer | off |
| `scripts/opsd_correct_think.sh` | verified integer answer | on |
| `scripts/opsd_pi_nothink.sh` | fixed wrong answer `π` | off |
| `scripts/opsd_pi_think.sh` | fixed wrong answer `π` | on |
| `scripts/opsd_instruction_nothink.sh` | detailed instruction, no answer | off |
| `scripts/opsd_instruction_think.sh` | detailed instruction, no answer | on |

Student and teacher use the Qwen3 chat template with the same explicit `enable_thinking` value. In the instruction-shift variants, the student is instructed to be concise and the teacher to be detailed.

## Training configuration

- Model: `/gpfs/share/home/2501210611/labShare/2501210611/model/qwen3-4b`
- 4×A800, DeepSpeed ZeRO-3, full student parameter updates
- Frozen full-model teacher initialized from step 0
- Per-device batch 1, gradient accumulation 4, global batch 16
- Prompt cap 1024, response cap 8192
- 100 optimizer steps; full checkpoints at 25, 50, 75, and 100
- AdamW, learning rate `5e-6`, cosine decay, 10% warmup, bf16
- vLLM colocate, TP=1 per rank, utilization `0.45`, sleep mode enabled
- Training attention: PyTorch SDPA; vLLM backend: XFormers; no FlashAttention execution
- WandB offline plus Trainer/Slurm logs
- No task reward, correctness signal, advantage, or training-time validation

Logged metrics include loss, student/teacher entropy, sampled-token log-probabilities, forward/reverse KL diagnostics, valid response tokens, generated length, rollout throughput, learning rate, gradient norm, and CUDA allocated/reserved/peak memory.

## Submit

Set the server dataset path and submit one experiment:

```bash
export DATASET_PATH=/server/path/to/dapo-train.parquet
sbatch scripts/opsd_correct_nothink.sh
```

Optional overrides are `OUTPUT_ROOT`, `WANDB_DIR`, `HF_HOME`, and `MASTER_PORT`.

Before the first training job, verify the existing `anchor` environment without installing anything:

```bash
bash scripts/smoke_test_imports.sh
```

## Evaluation

The evaluator is adapted from CAST and directly loads a full checkpoint. It covers AIME24, AIME25, AIME26, HMMT25, and MATH500 with pass@1/4/8/16.

```bash
export CHECKPOINT_PATH=/path/to/opsd_correct_nothink/checkpoint-100
export EVAL_DATA_ROOT=/server/path/to/eval/data
sbatch scripts/eval_nothink.sh
```

Use `scripts/eval_think.sh` for a thinking checkpoint. No LoRA adapter argument is used.

## Layout

- `src/opsd_trainer.py`: official OPSD trainer adapted for an independent frozen full model and TRL 0.22.1
- `src/data_collator.py`: three privileged-context constructions and Qwen3 chat templates
- `src/train_opsd.py`: DAPO loading, prompt filtering, trainer setup, and metrics
- `configs/`: ZeRO-3 and Accelerate configuration
- `eval/`: CAST-derived full-model math evaluation
- `vendor/verl`: safely extracted CAST `verl.zip`
- `vendor/OPSD_official`: upstream OPSD reference snapshot
- `vendor/trl_v0.22.1`: server-version API reference used for static compatibility checks
