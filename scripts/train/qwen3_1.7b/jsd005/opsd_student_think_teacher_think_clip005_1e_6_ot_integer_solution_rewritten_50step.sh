#!/bin/bash
#SBATCH --job-name=st_tt_clip005_1e6_ot_int_sol_rw_1p7b
#SBATCH --output=log/train/1.7b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=3:00:00
set -euo pipefail

# Qwen3-1.7B OPSD st_tt: full solution with rewritten final boxed answer.
BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}}
export RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_ot_integer_solution_rewritten_50step_1p7b}
export DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.opsd.solution.sthink_tthink.integer_answer.rewritten_frac_decimal.sampler_seed1.n3200.parquet}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_1p7b_fullparam_50step_ot_integer_solution_rewritten}
bash "${BASE_DIR}/scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_solution_50step.sh"
