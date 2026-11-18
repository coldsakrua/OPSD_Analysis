#!/bin/bash
#SBATCH --exclude=gpua800n03,gpua800n10,gpua800n13,gpua800n21
#SBATCH --job-name=st_tt_clip005_1e6_ot_int_sol_wr50_olmo7bt
#SBATCH --output=log/train/olmo3-7b-think/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=03:00:00
set -euo pipefail

# Olmo-3-7B-Think OPSD st_tt: full solution with wrong_rand50 final boxed answer.
BASE_DIR=${BASE_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}}
export RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_ot_integer_solution_wrong_rand50_50step_olmo7bt}
export DATASET_PATH=${DATASET_PATH:-${BASE_DIR}/data/openthoughts/preprocessed/openthoughts.opsd.solution.sthink_tthink.integer_answer.wrong_rand50.sampler_seed1.n3200.parquet}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-olmo3_7b_think_fullparam_50step_ot_integer_solution_wrong_rand50}
bash "${BASE_DIR}/scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_ot_integer_solution_50step.sh"
