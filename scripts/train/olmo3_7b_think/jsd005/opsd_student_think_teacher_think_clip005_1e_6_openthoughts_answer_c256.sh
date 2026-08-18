#!/bin/bash
#SBATCH --job-name=st_tt_clip005_1e6_ota_olmo7bt_c256
#SBATCH --output=log/train/olmo3-7b-think/opsd_%x.%j.out
#SBATCH --partition=GPUA800,GPUA800S,GPUA800L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --time=03:00:00
set -euo pipefail

# OTA (answer): same data/teacher as parent (max_prompt=1024), only shrink student rollout
# to max_completion=256 (aligned with cotlen easy/hard c256 wrappers).
# Keep samples/update = 64 (micro=4, gas=4, 4gpu); max_steps=100.
export JSD_TOKEN_CLIP=0.05
export MAX_PROMPT_LENGTH=1024
export MAX_COMPLETION_LENGTH=256
export PER_DEVICE_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=4
export MAX_STEPS=100
export SAVE_STEPS=25
export RUN_NAME=st_tt_clip005_1e_6_openthoughts_answer_olmo7bt_c256
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-olmo3_7b_think_fullparam_100step_openthoughts_answer_c256}
export BASE_DIR="${BASE_DIR:-${SLURM_SUBMIT_DIR:-/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis}}"

# Absolute path: Slurm copies this wrapper to /var/spool, so relative paths break.
bash /gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis/scripts/train/olmo3_7b_think/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_answer.sh
