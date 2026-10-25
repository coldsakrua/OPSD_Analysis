# OPSD code evaluation scripts

Self-contained under `OPSD_Analysis` (vendored judges + local data).

## Split: GPU generate → CPU judge

Same pattern as LiveCodeBench:

1. **GPU job** (`*_think.sh` / `*_nothink.sh`): generate only (`--skip-evaluate`), write `*.generations.json`
2. Auto-submit **CPU judge** on `C64M256G,C64M512G` (set `AUTO_JUDGE=0` to skip)
3. **CPU job** (`*_judge.sh`): official unit tests → metrics / results

| Dataset | Data | Generate | Judge |
|---|---|---|---|
| LCB v6 | `data/livecodebench/test6.jsonl` | `eval_lcb_{vllm,sglang}_local.py` | `eval_lcb_judge_local.py` |
| HumanEval+ | `data/humanevalplus/HumanEvalPlus.jsonl` | `eval_evalplus_{vllm,sglang}_local.py` | `eval_evalplus_judge_local.py` |
| MBPP+ | `data/mbppplus/MbppPlus.jsonl` | same | same |

Sampling / judge (aligned with current LCB scripts):

- Generate: **val-n=4**, **pass@1,4**
- CPU judge: partition **C64M512G**, **48 CPUs**, **240G**, `NUM_PROCESS=48`

## Example

```bash
cd OPSD_Analysis
# GPU generate + auto CPU judge
sbatch scripts/eval_code/1.7b/humanevalplus_think.sh
sbatch scripts/eval_code/8b/mbppplus_think.sh
sbatch scripts/eval_code/4b_instruct/humanevalplus_nothink.sh
sbatch scripts/eval_code/olmo3-7b-it/humanevalplus_sgl_nothink.sh

# Manual CPU judge only
sbatch scripts/eval_code/1.7b/humanevalplus_think_judge.sh
# or
EVALPLUS_GENERATIONS_JSON=... EVALPLUS_OUTPUT_JSON=... EVALPLUS_DATASET=humanevalplus \
  sbatch scripts/eval_code/run_evalplus_judge.sh

# Smoke (1 problem generate + local judge in one GPU job)
sbatch scripts/eval_code/1.7b/smoke_humanevalplus_think.sh
```

Helpers: `lcb_auto_submit_judge.sh`, `evalplus_auto_submit_judge.sh`.
