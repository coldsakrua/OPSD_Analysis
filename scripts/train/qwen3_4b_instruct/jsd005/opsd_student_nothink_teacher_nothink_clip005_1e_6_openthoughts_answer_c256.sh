#!/bin/bash
#SBATCH --job-name=snt_tnt_clip005_1e6_otai_c256
#SBATCH --output=log/train/4b-instruct/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=72:00:00
#SBATCH --exclude=gpua800n02,gpua800n03,gpua800n04,gpua800n05,gpua800n06,gpua800n09,gpua800n10,gpua800n13,gpua800n14
set -euo pipefail

# OTA (answer): same data/teacher as parent (max_prompt=1024), only shrink student rollout
# to max_completion=256 (aligned with cotlen easy/hard c256 wrappers).
# Keep samples/update = 32 (micro=4, gas=4, 2gpu); max_steps=100.
export JSD_TOKEN_CLIP=0.05
export MAX_PROMPT_LENGTH=1024
export MAX_COMPLETION_LENGTH=256
export PER_DEVICE_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=4
export MAX_STEPS=100
export SAVE_STEPS=25
export RUN_NAME=snt_tnt_clip005_1e_6_openthoughts_answer_instruct_c256
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-qwen3_4b_instruct_fullparam_100step_openthoughts_c256}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/qwen3_4b_instruct/jsd005/opsd_student_nothink_teacher_nothink_clip005_1e_6_openthoughts_answer.sh
