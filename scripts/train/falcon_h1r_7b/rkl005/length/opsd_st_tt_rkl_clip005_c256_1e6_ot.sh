#!/bin/bash
#SBATCH --job-name=st_tt_rkl_clip005_c256_falcon7b
#SBATCH --output=log/train/falcon_h1r_7b/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=12:00:00
set -euo pipefail

# Falcon-H1R-7B st_tt + reverse KL (beta=1.0) + clip=0.05: max_completion=256.
# Match eval stack (conda falcon / SGLang triton+pytorch / deepseek-r1 reasoning parser).
# Keep samples/update = 32 (baseline micro=2,gas=4,4gpu):
#   micro=2, gas=4 → gbs=32; max_steps=100 → 3200 samples total.
export JSD_TOKEN_CLIP=${JSD_TOKEN_CLIP:-0.05}
export BETA=${BETA:-1.0}
export MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-256}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
export TARGET_GLOBAL_BATCH=${TARGET_GLOBAL_BATCH:-32}
export MAX_STEPS=${MAX_STEPS:-100}
export SAVE_STEPS=${SAVE_STEPS:-25}
export RUN_NAME=${RUN_NAME:-st_tt_rkl_clip005_1e_6_ot_falcon7b_c256}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-falcon_h1r_7b_st_tt_rkl_clip005_longgen_c256}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/falcon_h1r_7b/rkl005/opsd_student_think_teacher_think_rkl_clip005_1e_6_openthoughts.sh
