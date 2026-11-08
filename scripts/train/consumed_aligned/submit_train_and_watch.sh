#!/bin/bash
# Build the consumed-aligned parquets on a CPU node, then submit 10 trains
# and one watcher. Trains start only after the parquet job succeeds.
#
#   bash scripts/train/consumed_aligned/submit_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

PREP="scripts/data/preprocess_consumed_aligned_answer_unrelated.sh"
TRAIN="scripts/train/consumed_aligned/train_one.sh"
WATCH="scripts/train/consumed_aligned/sbatch_watch.sh"
OUT_DIR="${ROOT}/log/train/consumed_aligned"
mkdir -p "${OUT_DIR}" log/data \
  log/train/1.7b log/train/4b log/train/4b-instruct \
  log/train/olmo3-7b-think log/train/olmo3-7b-instruct

chmod +x "${PREP}" "${TRAIN}" "${WATCH}" \
  scripts/data/build_consumed_aligned_answer_unrelated.py \
  scripts/train/consumed_aligned/watch_then_eval.sh

prep_jid=$(sbatch --parsable --chdir="${ROOT}" "${PREP}")
prep_jid=${prep_jid%%;*}
echo "[submit] preprocess -> jid=${prep_jid}"

submit_one() {
  local spec=$1 job_name=$2 gres=$3 cpus=$4 mem=$5 time_limit=$6 exclude=$7 log_path=$8
  local jid
  jid=$(sbatch --parsable --chdir="${ROOT}" \
    --dependency="afterok:${prep_jid}" \
    --job-name="${job_name}" \
    --gres="${gres}" \
    --cpus-per-task="${cpus}" \
    --mem="${mem}" \
    --time="${time_limit}" \
    --exclude="${exclude}" \
    --output="${log_path}" \
    --export=ALL,SPEC="${spec}",BASE_DIR="${ROOT}" \
    "${TRAIN}")
  jid=${jid%%;*}
  echo "[submit] ${spec} -> jid=${jid}" >&2
  printf '%s\n' "${jid}"
}

# spec job_name gres cpus mem time exclude log
# 4B reads the first 3200 rows of the shared 6400-row Qwen think file.
declare -a JIDS=()
JIDS+=("$(submit_one qwen17_answer      aln_ans_1p7b  gpu:2 14 220G 12:00:00 gpua800n13 \
  log/train/1.7b/opsd_%x.%j.out)")
JIDS+=("$(submit_one qwen17_unrelated   aln_ios_1p7b  gpu:2 14 220G 12:00:00 gpua800n13 \
  log/train/1.7b/opsd_%x.%j.out)")
JIDS+=("$(submit_one qwen4b_answer       aln_ans_4b    gpu:2 14 220G 24:00:00 gpua800n03,gpua800n09,gpua800n13 \
  log/train/4b/opsd_%x.%j.out)")
JIDS+=("$(submit_one qwen4b_unrelated    aln_ios_4b    gpu:2 14 220G 24:00:00 gpua800n03,gpua800n09,gpua800n13 \
  log/train/4b/opsd_%x.%j.out)")
JIDS+=("$(submit_one qwen4bi_answer      aln_ans_4bi   gpu:2 14 220G 24:00:00 \
  gpua800n02,gpua800n03,gpua800n04,gpua800n05,gpua800n06,gpua800n09,gpua800n10,gpua800n13,gpua800n14 \
  log/train/4b-instruct/opsd_%x.%j.out)")
JIDS+=("$(submit_one qwen4bi_unrelated   aln_ios_4bi   gpu:2 14 220G 24:00:00 \
  gpua800n02,gpua800n03,gpua800n04,gpua800n05,gpua800n06,gpua800n09,gpua800n10,gpua800n13,gpua800n14 \
  log/train/4b-instruct/opsd_%x.%j.out)")
JIDS+=("$(submit_one olmo_think_answer   aln_ans_olmo  gpu:4 28 400G 12:00:00 gpua800n03,gpua800n10,gpua800n13,gpua800n21 \
  log/train/olmo3-7b-think/opsd_%x.%j.out)")
JIDS+=("$(submit_one olmo_think_unrelated aln_ios_olmo gpu:4 28 400G 12:00:00 gpua800n03,gpua800n10,gpua800n13,gpua800n21 \
  log/train/olmo3-7b-think/opsd_%x.%j.out)")
JIDS+=("$(submit_one olmo_it_answer      aln_ans_olmobi gpu:4 28 400G 12:00:00 gpua800n03,gpua800n10,gpua800n13,gpua800n21 \
  log/train/olmo3-7b-instruct/opsd_%x.%j.out)")
JIDS+=("$(submit_one olmo_it_unrelated   aln_ios_olmobi gpu:4 28 400G 12:00:00 gpua800n03,gpua800n10,gpua800n13,gpua800n21 \
  log/train/olmo3-7b-instruct/opsd_%x.%j.out)")

MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"
{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript\teval_tag"
  echo -e "qwen3_1.7b\tqwen3_1.7b\tst_tt_clip005_1e_6_ot_ans_same6400_1p7b\t${JIDS[0]}\t${TRAIN}\tst_tt_ot_ans_same6400_1p7b_ckpt100"
  echo -e "qwen3_1.7b\tqwen3_1.7b\tst_tt_clip005_1e_6_ot_ios_same6400_1p7b\t${JIDS[1]}\t${TRAIN}\tst_tt_ot_ios_same6400_1p7b_ckpt100"
  echo -e "qwen3_4b\tqwen3_4b\tst_tt_clip005_1e_6_ot_ans_same3200_4b\t${JIDS[2]}\t${TRAIN}\tst_tt_ot_ans_same3200_4b_ckpt100"
  echo -e "qwen3_4b\tqwen3_4b\tst_tt_clip005_1e_6_ot_ios_same3200_4b\t${JIDS[3]}\t${TRAIN}\tst_tt_ot_ios_same3200_4b_ckpt100"
  echo -e "qwen3_4b_instruct\tqwen3_4b_instruct\tsnt_tnt_clip005_1e_6_ot_ans_same3200_instruct\t${JIDS[4]}\t${TRAIN}\tsnt_tnt_ot_ans_same3200_instruct_ckpt100"
  echo -e "qwen3_4b_instruct\tqwen3_4b_instruct\tsnt_tnt_clip005_1e_6_ot_ios_same3200_instruct\t${JIDS[5]}\t${TRAIN}\tsnt_tnt_ot_ios_same3200_instruct_ckpt100"
  echo -e "olmo3_7b_think\tolmo3_7b_think\tst_tt_clip005_1e_6_ot_ans_same6400_olmo7bt\t${JIDS[6]}\t${TRAIN}\tst_tt_ot_ans_same6400_olmo7bt_ckpt100"
  echo -e "olmo3_7b_think\tolmo3_7b_think\tst_tt_clip005_1e_6_ot_ios_same6400_olmo7bt\t${JIDS[7]}\t${TRAIN}\tst_tt_ot_ios_same6400_olmo7bt_ckpt100"
  echo -e "olmo3_7b_instruct\tolmo3_7b_instruct\tsnt_tnt_clip005_1e_6_ot_ans_same6400_olmo7bit\t${JIDS[8]}\t${TRAIN}\tsnt_tnt_ot_ans_same6400_olmo7bit_ckpt100"
  echo -e "olmo3_7b_instruct\tolmo3_7b_instruct\tsnt_tnt_clip005_1e_6_ot_ios_same6400_olmo7bit\t${JIDS[9]}\t${TRAIN}\tsnt_tnt_ot_ios_same6400_olmo7bit_ckpt100"
} >"${MANIFEST_STAMPED}"
cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --dependency="afterok:${prep_jid}" \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${WATCH}")
watch_jid=${watch_jid%%;*}
echo "[submit] watch -> jid=${watch_jid} (afterok:${prep_jid})"
echo "[submit] done prep=${prep_jid}"
cat "${MANIFEST}"
