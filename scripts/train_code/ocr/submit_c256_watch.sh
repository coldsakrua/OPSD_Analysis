#!/bin/bash
# Attach a CPU-node watcher to the 6 OCR c256 trains (max_completion=256).
# When checkpoint-100 is ready, submit LCB / HumanEval+ / MBPP+ generate jobs.
# Generate scripts auto-submit CPU judges.
#
# Eval logs stay in the same dirs as the 1024 OCR runs, but job-name/log prefix
# is ocr_c256_* so they are distinct from ocr_qwen3_1.7b_* (1024) and mb_1.7b_th
# (base model). Example:
#   log/eval_code/1.7b/lcb_v6/think/ocr_c256_qwen3_1.7b_lcb_th.<jid>.out
#   log/eval_code/1.7b/lcb_v6/think/ocr_c256_qwen3_1.7b_lcb_j.<jid>.out
#
# Usage (from OPSD_Analysis):
#   bash scripts/train_code/ocr/submit_c256_watch.sh
# Override train jids:
#   TRAIN_1P7B=3940537 TRAIN_4B=3940538 ... bash scripts/train_code/ocr/submit_c256_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train_code/ocr_c256"
mkdir -p "${OUT_DIR}"
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

TRAIN_1P7B=${TRAIN_1P7B:-3940537}
TRAIN_4B=${TRAIN_4B:-3940538}
TRAIN_8B=${TRAIN_8B:-3940539}
TRAIN_4BI=${TRAIN_4BI:-3940540}
TRAIN_OLMO_T=${TRAIN_OLMO_T:-3940541}
TRAIN_OLMO_I=${TRAIN_OLMO_I:-3940542}

# model_key | model_tag | run_name | train_jid | script | eval_tag
declare -a JOBS=(
  "qwen3_1.7b|qwen3_1.7b|st_tt_clip005_1e_6_opencodereasoning_1p7b_c256|${TRAIN_1P7B}|scripts/train_code/qwen3_1.7b/jsd005/length/opsd_st_tt_clip005_c256_1e6_ocr.sh|ocr_c256_qwen3_1.7b_ckpt100"
  "qwen3_4b|qwen3_4b|st_tt_clip005_1e_6_opencodereasoning_4b_c256|${TRAIN_4B}|scripts/train_code/qwen3_4b/jsd005/length/opsd_st_tt_clip005_c256_1e6_ocr.sh|ocr_c256_qwen3_4b_ckpt100"
  "qwen3_8b|qwen3_8b|st_tt_clip005_1e_6_opencodereasoning_8b_c256|${TRAIN_8B}|scripts/train_code/qwen3_8b/jsd005/length/opsd_st_tt_clip005_c256_1e6_ocr.sh|ocr_c256_qwen3_8b_ckpt100"
  "qwen3_4b_instruct|qwen3_4b_instruct|snt_tnt_clip005_1e_6_opencodereasoning_instruct_c256|${TRAIN_4BI}|scripts/train_code/qwen3_4b_instruct/jsd005/length/opsd_snt_tnt_clip005_c256_1e6_ocr.sh|ocr_c256_qwen3_4b_instruct_ckpt100"
  "olmo3_7b_think|olmo3_7b_think|st_tt_clip005_1e_6_opencodereasoning_olmo7bt_c256|${TRAIN_OLMO_T}|scripts/train_code/olmo3_7b_think/jsd005/length/opsd_st_tt_clip005_c256_1e6_ocr.sh|ocr_c256_olmo3_7b_think_ckpt100"
  "olmo3_7b_instruct|olmo3_7b_instruct|snt_tnt_clip005_1e_6_opencodereasoning_olmo7bit_c256|${TRAIN_OLMO_I}|scripts/train_code/olmo3_7b_instruct/jsd005/length/opsd_snt_tnt_clip005_c256_1e6_ocr.sh|ocr_c256_olmo3_7b_instruct_ckpt100"
)

chmod +x \
  scripts/train_code/ocr/sbatch_watch.sh \
  scripts/train_code/ocr/watch_then_eval.sh \
  scripts/train_code/ocr/submit_three_code_evals.sh

{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript\teval_tag"
} >"${MANIFEST_STAMPED}"

echo "[c256-watch] root=${ROOT}"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r model_key model_tag run_name train_jid script eval_tag <<<"${entry}"
  if [[ -z "${train_jid}" || "${train_jid}" == "REPLACE_ME" ]]; then
    echo "[error] missing train_jid for ${model_key}" >&2
    exit 1
  fi
  echo -e "${model_key}\t${model_tag}\t${run_name}\t${train_jid}\t${script}\t${eval_tag}" >>"${MANIFEST_STAMPED}"
  echo "[c256-watch] ${eval_tag} train_jid=${train_jid}"
done

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[c256-watch] manifest=${MANIFEST}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_ocr_c256 \
  --output=log/train_code/ocr_c256/watch_slurm.%j.out \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}",CKPT_STEP=100 \
  "${ROOT}/scripts/train_code/ocr/sbatch_watch.sh")
echo "[c256-watch] watch -> jid=${watch_jid}"
echo "[c256-watch] generate logs: log/eval_code/<model>/{lcb_v6,humanevalplus,mbppplus}/.../ocr_c256_<model>_{lcb,he,mb}_{th|nt}.<jid>.out"
echo "[c256-watch] judge logs:     same dir, suffix _j instead of _th/_nt"
cat "${MANIFEST}"
