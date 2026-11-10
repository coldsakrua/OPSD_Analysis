#!/bin/bash
# Submit the 10 aligned answer-only / unrelated trains at max_completion=256,
# plus one watcher. Reuses the c1024 consumption-order parquets. Full-solution
# c256 already keeps the same global batch and 100 steps, so the consumed
# problems stay the same.
#
#   bash scripts/train/consumed_aligned/submit_c256_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

TRAIN="scripts/train/consumed_aligned/train_one.sh"
WATCH="scripts/train/consumed_aligned/sbatch_watch.sh"
OUT_DIR="${ROOT}/log/train/consumed_aligned"
mkdir -p "${OUT_DIR}" \
  log/train/1.7b log/train/4b log/train/4b-instruct \
  log/train/olmo3-7b-think log/train/olmo3-7b-instruct

chmod +x "${TRAIN}" "${WATCH}" scripts/train/consumed_aligned/watch_then_eval.sh

submit_one() {
  local spec=$1 job_name=$2 gres=$3 cpus=$4 mem=$5 time_limit=$6 exclude=$7 log_path=$8
  local jid
  jid=$(sbatch --parsable --chdir="${ROOT}" \
    --job-name="${job_name}" \
    --gres="${gres}" \
    --cpus-per-task="${cpus}" \
    --mem="${mem}" \
    --time="${time_limit}" \
    --exclude="${exclude}" \
    --output="${log_path}" \
    --export=ALL,SPEC="${spec}",COMPLETION_LENGTH=256,BASE_DIR="${ROOT}" \
    "${TRAIN}")
  jid=${jid%%;*}
  echo "[submit-c256] ${spec} -> jid=${jid}" >&2
  printf '%s\n' "${jid}"
}

declare -a JIDS=()
JIDS+=("$(submit_one qwen17_answer      aln256_ans_1p7b  gpu:2 14 220G 12:00:00 gpua800n13 \
  log/train/1.7b/opsd_%x.%j.out)")
JIDS+=("$(submit_one qwen17_unrelated   aln256_ios_1p7b  gpu:2 14 220G 12:00:00 gpua800n13 \
  log/train/1.7b/opsd_%x.%j.out)")
JIDS+=("$(submit_one qwen4b_answer       aln256_ans_4b    gpu:2 14 220G 12:00:00 gpua800n03,gpua800n09,gpua800n13 \
  log/train/4b/opsd_%x.%j.out)")
JIDS+=("$(submit_one qwen4b_unrelated    aln256_ios_4b    gpu:2 14 220G 12:00:00 gpua800n03,gpua800n09,gpua800n13 \
  log/train/4b/opsd_%x.%j.out)")
JIDS+=("$(submit_one qwen4bi_answer      aln256_ans_4bi   gpu:2 14 220G 12:00:00 \
  gpua800n02,gpua800n03,gpua800n04,gpua800n05,gpua800n06,gpua800n09,gpua800n10,gpua800n13,gpua800n14 \
  log/train/4b-instruct/opsd_%x.%j.out)")
JIDS+=("$(submit_one qwen4bi_unrelated   aln256_ios_4bi   gpu:2 14 220G 12:00:00 \
  gpua800n02,gpua800n03,gpua800n04,gpua800n05,gpua800n06,gpua800n09,gpua800n10,gpua800n13,gpua800n14 \
  log/train/4b-instruct/opsd_%x.%j.out)")
JIDS+=("$(submit_one olmo_think_answer   aln256_ans_olmo  gpu:4 28 400G 12:00:00 gpua800n03,gpua800n10,gpua800n13,gpua800n21 \
  log/train/olmo3-7b-think/opsd_%x.%j.out)")
JIDS+=("$(submit_one olmo_think_unrelated aln256_ios_olmo gpu:4 28 400G 12:00:00 gpua800n03,gpua800n10,gpua800n13,gpua800n21 \
  log/train/olmo3-7b-think/opsd_%x.%j.out)")
JIDS+=("$(submit_one olmo_it_answer      aln256_ans_olmobi gpu:4 28 400G 12:00:00 gpua800n03,gpua800n10,gpua800n13,gpua800n21 \
  log/train/olmo3-7b-instruct/opsd_%x.%j.out)")
JIDS+=("$(submit_one olmo_it_unrelated   aln256_ios_olmobi gpu:4 28 400G 12:00:00 gpua800n03,gpua800n10,gpua800n13,gpua800n21 \
  log/train/olmo3-7b-instruct/opsd_%x.%j.out)")

MANIFEST="${OUT_DIR}/submit_c256_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_c256_${STAMP}.tsv"
{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript\teval_tag"
  echo -e "qwen3_1.7b\tqwen3_1.7b\tst_tt_clip005_1e_6_ot_ans_same6400_1p7b_c256\t${JIDS[0]}\t${TRAIN}\tst_tt_ot_ans_same6400_1p7b_c256_ckpt100"
  echo -e "qwen3_1.7b\tqwen3_1.7b\tst_tt_clip005_1e_6_ot_ios_same6400_1p7b_c256\t${JIDS[1]}\t${TRAIN}\tst_tt_ot_ios_same6400_1p7b_c256_ckpt100"
  echo -e "qwen3_4b\tqwen3_4b\tst_tt_clip005_1e_6_ot_ans_same3200_4b_c256\t${JIDS[2]}\t${TRAIN}\tst_tt_ot_ans_same3200_4b_c256_ckpt100"
  echo -e "qwen3_4b\tqwen3_4b\tst_tt_clip005_1e_6_ot_ios_same3200_4b_c256\t${JIDS[3]}\t${TRAIN}\tst_tt_ot_ios_same3200_4b_c256_ckpt100"
  echo -e "qwen3_4b_instruct\tqwen3_4b_instruct\tsnt_tnt_clip005_1e_6_ot_ans_same3200_instruct_c256\t${JIDS[4]}\t${TRAIN}\tsnt_tnt_ot_ans_same3200_instruct_c256_ckpt100"
  echo -e "qwen3_4b_instruct\tqwen3_4b_instruct\tsnt_tnt_clip005_1e_6_ot_ios_same3200_instruct_c256\t${JIDS[5]}\t${TRAIN}\tsnt_tnt_ot_ios_same3200_instruct_c256_ckpt100"
  echo -e "olmo3_7b_think\tolmo3_7b_think\tst_tt_clip005_1e_6_ot_ans_same6400_olmo7bt_c256\t${JIDS[6]}\t${TRAIN}\tst_tt_ot_ans_same6400_olmo7bt_c256_ckpt100"
  echo -e "olmo3_7b_think\tolmo3_7b_think\tst_tt_clip005_1e_6_ot_ios_same6400_olmo7bt_c256\t${JIDS[7]}\t${TRAIN}\tst_tt_ot_ios_same6400_olmo7bt_c256_ckpt100"
  echo -e "olmo3_7b_instruct\tolmo3_7b_instruct\tsnt_tnt_clip005_1e_6_ot_ans_same6400_olmo7bit_c256\t${JIDS[8]}\t${TRAIN}\tsnt_tnt_ot_ans_same6400_olmo7bit_c256_ckpt100"
  echo -e "olmo3_7b_instruct\tolmo3_7b_instruct\tsnt_tnt_clip005_1e_6_ot_ios_same6400_olmo7bit_c256\t${JIDS[9]}\t${TRAIN}\tsnt_tnt_ot_ios_same6400_olmo7bit_c256_ckpt100"
} >"${MANIFEST_STAMPED}"
cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit-c256] manifest=${MANIFEST}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_ot_c256 \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${WATCH}")
watch_jid=${watch_jid%%;*}
echo "[submit-c256] watch -> jid=${watch_jid}"
echo "[submit-c256] done"
cat "${MANIFEST}"
