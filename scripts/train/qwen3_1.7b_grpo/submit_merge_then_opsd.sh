#!/bin/bash
# Merge GRPO@100 FSDP → hf_merged, then submit OpenThoughts OPSD (same recipe as 4b_grpo).
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/qwen3_1.7b_grpo/submit_merge_then_opsd.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

MERGE_SCRIPT=scripts/sft_rl/merge_gs100_1p7b_sft15000_4706799.sh
TRAIN_SCRIPT=scripts/train/qwen3_1.7b_grpo/jsd005/opsd_student_think_teacher_think_clip005_1e_6_openthoughts_grpo.sh
ACTOR="${ROOT}/outputs/qwen3_1.7b_base/grpo_think_1p7b_sft15000_4gpu_nooverlong/4706799/global_step_100/actor"
MERGED="${ACTOR}/hf_merged"

chmod +x "${MERGE_SCRIPT}" "${TRAIN_SCRIPT}" scripts/train/qwen3_1.7b_grpo/submit_merge_then_opsd.sh

if [[ ! -d "${ACTOR}" ]]; then
  echo "[error] missing GRPO actor: ${ACTOR}" >&2
  exit 1
fi

hf_ok() {
  local p=$1
  [[ -f "${p}/config.json" ]] || return 1
  [[ -f "${p}/model.safetensors" ]] \
    || [[ -f "${p}/model.safetensors.index.json" ]] \
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

WATCH_SCRIPT=scripts/train/qwen3_1.7b_grpo/watch_then_eval.sh
RUN_NAME=${RUN_NAME:-st_tt_clip005_1e_6_openthoughts_1p7b_grpo}
CKPT_STEP=${CKPT_STEP:-100}
chmod +x "${WATCH_SCRIPT}"

if hf_ok "${MERGED}"; then
  echo "[submit] hf_merged already ready: ${MERGED}"
  train_jid=$(sbatch --parsable --chdir="${ROOT}" "${TRAIN_SCRIPT}")
  echo "[submit] opsd train_jid=${train_jid} (no merge needed)"
else
  merge_jid=$(sbatch --parsable --chdir="${ROOT}" "${MERGE_SCRIPT}")
  echo "[submit] merge_jid=${merge_jid}"
  train_jid=$(sbatch --parsable --chdir="${ROOT}" --dependency=afterok:${merge_jid} "${TRAIN_SCRIPT}")
  echo "[submit] opsd train_jid=${train_jid} (afterok:${merge_jid})"
fi

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --export=ALL,BASE_DIR="${ROOT}",TRAIN_JID="${train_jid}",RUN_NAME="${RUN_NAME}",MODEL_TAG=qwen3_1.7b_base,CKPT_STEP="${CKPT_STEP}" \
  "${WATCH_SCRIPT}")
echo "[submit] watch_jid=${watch_jid} → aime24/25/26 + hmmt25 @ checkpoint-${CKPT_STEP}"

echo "[submit] init model → ${MERGED}"
echo "[submit] train script → ${TRAIN_SCRIPT}"
echo "[submit] log → log/train/1.7b_grpo/opsd_st_tt_clip005_1e6_1p7bg.${train_jid}.out"
echo "[submit] watch log → log/train/1.7b_grpo/watch_slurm.${watch_jid}.out"
echo "[submit] out → outputs/qwen3_1.7b_base/${RUN_NAME}/${train_jid}/"
