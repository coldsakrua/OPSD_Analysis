#!/bin/bash
# Train seeds 1024 / 65536 for answer-only, unrelated, and cot+solution.
# - ans / ios: same6400 consumed-aligned, max_completion=256, SequentialSampler
# - cot: same full filtered cot parquet as seed-42 cot runs
# Watcher submits think evals with SEED=42 only.
#
# Order: submit all Qwen3-1.7B jobs first, then Olmo-3-7B-Think.
#
#   bash scripts/train/consumed_aligned/submit_ans_ios_cot_train_seeds.sh
#   MODEL=qwen3_1.7b bash scripts/train/consumed_aligned/submit_ans_ios_cot_train_seeds.sh
#   MODEL=olmo3_7b_think bash scripts/train/consumed_aligned/submit_ans_ios_cot_train_seeds.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

MODEL=${MODEL:-all}
TRAIN_ONE="scripts/train/consumed_aligned/train_one.sh"
WATCH="scripts/train/consumed_aligned/sbatch_watch.sh"
OUT_DIR="${ROOT}/log/train/consumed_aligned"
mkdir -p "${OUT_DIR}" \
  log/train/1.7b log/train/olmo3-7b-think \
  scripts/train/qwen3_1.7b/jsd005/seed \
  scripts/train/olmo3_7b_think/jsd005/seed

chmod +x \
  "${TRAIN_ONE}" "${WATCH}" \
  scripts/train/consumed_aligned/watch_then_eval.sh \
  scripts/train/qwen3_1.7b/jsd005/seed/opsd_st_tt_clip005_cot_t4096_seed1024_ot.sh \
  scripts/train/qwen3_1.7b/jsd005/seed/opsd_st_tt_clip005_cot_t4096_seed65536_ot.sh \
  scripts/train/olmo3_7b_think/jsd005/seed/opsd_st_tt_clip005_cot_seed1024_ot.sh \
  scripts/train/olmo3_7b_think/jsd005/seed/opsd_st_tt_clip005_cot_seed65536_ot.sh

STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_ans_ios_cot_seeds_${STAMP}.tsv"
MANIFEST_LATEST="${OUT_DIR}/submit_ans_ios_cot_seeds_latest.tsv"
: >"${MANIFEST_STAMPED}"
echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript\teval_tag" >>"${MANIFEST_STAMPED}"

submit_aligned() {
  local spec=$1 job_name=$2 seed=$3 gres=$4 cpus=$5 mem=$6 time_limit=$7 exclude=$8 log_path=$9
  local run_name=${10} eval_tag=${11} model_key=${12} model_tag=${13}
  local jid
  jid=$(sbatch --parsable --chdir="${ROOT}" \
    --job-name="${job_name}" \
    --gres="${gres}" \
    --cpus-per-task="${cpus}" \
    --mem="${mem}" \
    --time="${time_limit}" \
    --exclude="${exclude}" \
    --output="${log_path}" \
    --export=ALL,SPEC="${spec}",COMPLETION_LENGTH=256,SEED="${seed}",BASE_DIR="${ROOT}" \
    "${TRAIN_ONE}")
  jid=${jid%%;*}
  echo "[submit-seed] ${spec} seed=${seed} -> jid=${jid} run=${run_name}" >&2
  echo -e "${model_key}\t${model_tag}\t${run_name}\t${jid}\t${TRAIN_ONE}\t${eval_tag}" >>"${MANIFEST_STAMPED}"
}

submit_script() {
  local script=$1 run_name=$2 eval_tag=$3 model_key=$4 model_tag=$5
  local jid
  jid=$(sbatch --parsable --chdir="${ROOT}" --export=ALL,BASE_DIR="${ROOT}" "${script}")
  jid=${jid%%;*}
  echo "[submit-seed] ${script} -> jid=${jid} run=${run_name}" >&2
  echo -e "${model_key}\t${model_tag}\t${run_name}\t${jid}\t${script}\t${eval_tag}" >>"${MANIFEST_STAMPED}"
}

submit_qwen17() {
  local seed
  for seed in 1024 65536; do
    submit_aligned qwen17_answer "aln256_ans_s${seed}_1p7b" "${seed}" \
      gpu:2 14 220G 12:00:00 gpua800n13 log/train/1.7b/opsd_%x.%j.out \
      "st_tt_clip005_1e_6_ot_ans_same6400_1p7b_c256_seed${seed}" \
      "st_tt_ot_ans_same6400_1p7b_c256_seed${seed}_ckpt100" \
      qwen3_1.7b qwen3_1.7b
    submit_aligned qwen17_unrelated "aln256_ios_s${seed}_1p7b" "${seed}" \
      gpu:2 14 220G 12:00:00 gpua800n13 log/train/1.7b/opsd_%x.%j.out \
      "st_tt_clip005_1e_6_ot_ios_same6400_1p7b_c256_seed${seed}" \
      "st_tt_ot_ios_same6400_1p7b_c256_seed${seed}_ckpt100" \
      qwen3_1.7b qwen3_1.7b
    submit_script \
      "scripts/train/qwen3_1.7b/jsd005/seed/opsd_st_tt_clip005_cot_t4096_seed${seed}_ot.sh" \
      "st_tt_clip005_1e_6_openthoughts_cot_t4096_seed${seed}_1p7b" \
      "st_tt_ot_cot_t4096_seed${seed}_ckpt100" \
      qwen3_1.7b qwen3_1.7b
  done
}

submit_olmo() {
  local seed excl="gpua800n03,gpua800n10,gpua800n13,gpua800n21"
  for seed in 1024 65536; do
    submit_aligned olmo_think_answer "aln256_ans_s${seed}_olmo" "${seed}" \
      gpu:4 28 400G 12:00:00 "${excl}" log/train/olmo3-7b-think/opsd_%x.%j.out \
      "st_tt_clip005_1e_6_ot_ans_same6400_olmo7bt_c256_seed${seed}" \
      "st_tt_ot_ans_same6400_olmo7bt_c256_seed${seed}_ckpt100" \
      olmo3_7b_think olmo3_7b_think
    submit_aligned olmo_think_unrelated "aln256_ios_s${seed}_olmo" "${seed}" \
      gpu:4 28 400G 12:00:00 "${excl}" log/train/olmo3-7b-think/opsd_%x.%j.out \
      "st_tt_clip005_1e_6_ot_ios_same6400_olmo7bt_c256_seed${seed}" \
      "st_tt_ot_ios_same6400_olmo7bt_c256_seed${seed}_ckpt100" \
      olmo3_7b_think olmo3_7b_think
    submit_script \
      "scripts/train/olmo3_7b_think/jsd005/seed/opsd_st_tt_clip005_cot_seed${seed}_ot.sh" \
      "st_tt_clip005_1e_6_openthoughts_cot_seed${seed}_olmo7bt" \
      "st_tt_ot_cot_olmo7bt_seed${seed}_ckpt100" \
      olmo3_7b_think olmo3_7b_think
  done
}

do_qwen=0
do_olmo=0
case "${MODEL}" in
  all)
    do_qwen=1
    do_olmo=1
    ;;
  qwen3_1.7b|1.7b)
    do_qwen=1
    ;;
  olmo3_7b_think|olmo)
    do_olmo=1
    ;;
  *)
    echo "[error] unknown MODEL=${MODEL} (use all|qwen3_1.7b|olmo3_7b_think)" >&2
    exit 1
    ;;
esac

if [[ "${do_qwen}" -eq 1 ]]; then
  echo "[submit-seed] ==== Qwen3-1.7B (ans/ios same6400 c256 + cot) ===="
  submit_qwen17
fi
if [[ "${do_olmo}" -eq 1 ]]; then
  echo "[submit-seed] ==== Olmo-3-7B-Think (ans/ios same6400 c256 + cot) ===="
  submit_olmo
fi

cp -f "${MANIFEST_STAMPED}" "${MANIFEST_LATEST}"
echo "[submit-seed] manifest=${MANIFEST_LATEST}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_ans_ios_cot_seed \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST_LATEST}" \
  "${WATCH}")
watch_jid=${watch_jid%%;*}
echo "[submit-seed] watch -> jid=${watch_jid} (eval SEED=42)"
echo "[submit-seed] done"
cat "${MANIFEST_LATEST}"
