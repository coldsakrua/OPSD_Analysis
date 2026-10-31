#!/bin/bash
# Submit OCR OPSD trains (c256 + c1024) for:
#   deepseek_r1_1.5b, falcon_h1r_7b, mimo_7b_rl
# + one CPU watcher (ckpt-100 → 3 code evals each)
# + optional base-model code evals for all three.
#
# Usage (from OPSD_Analysis):
#   bash scripts/train_code/ocr/submit_r1_falcon_mimo_c256_c1024_train_and_watch.sh
# Skip base evals:
#   SUBMIT_BASE_EVAL=0 bash scripts/train_code/ocr/submit_r1_falcon_mimo_c256_c1024_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train_code/ocr_r1_falcon_mimo"
mkdir -p "${OUT_DIR}" \
  log/train_code/{deepseek_r1_1.5b,falcon_h1r_7b,mimo_7b_rl}
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

DATASET_PATH=${DATASET_PATH:-${ROOT}/data/opencodereasoning/preprocessed/ocr.opsd.solution.shared.qwen_olmo.tmax2048.parquet}
SUBMIT_BASE_EVAL=${SUBMIT_BASE_EVAL:-1}
MODEL_ROOT=${MODEL_ROOT:-/gpfs/share/home/2501210611/labShare/2501210611/model}

declare -a JOBS=(
  "deepseek_r1_1.5b|deepseek_r1_1.5b|st_tt_clip005_1e_6_opencodereasoning_r1_15b_c256|scripts/train_code/deepseek_r1_1.5b/jsd005/length/opsd_st_tt_clip005_c256_1e6_ocr.sh|ocr_c256_deepseek_r1_1.5b_ckpt100"
  "deepseek_r1_1.5b|deepseek_r1_1.5b|st_tt_clip005_1e_6_opencodereasoning_r1_15b_c1024|scripts/train_code/deepseek_r1_1.5b/jsd005/length/opsd_st_tt_clip005_c1024_1e6_ocr.sh|ocr_c1024_deepseek_r1_1.5b_ckpt100"
  "falcon_h1r_7b|falcon_h1r_7b|st_tt_clip005_1e_6_opencodereasoning_falcon7b_c256|scripts/train_code/falcon_h1r_7b/jsd005/length/opsd_st_tt_clip005_c256_1e6_ocr.sh|ocr_c256_falcon_h1r_7b_ckpt100"
  "falcon_h1r_7b|falcon_h1r_7b|st_tt_clip005_1e_6_opencodereasoning_falcon7b_c1024|scripts/train_code/falcon_h1r_7b/jsd005/length/opsd_st_tt_clip005_c1024_1e6_ocr.sh|ocr_c1024_falcon_h1r_7b_ckpt100"
  "mimo_7b_rl|mimo_7b_rl|st_tt_clip005_1e_6_opencodereasoning_mimo7b_c256|scripts/train_code/mimo_7b_rl/jsd005/length/opsd_st_tt_clip005_c256_1e6_ocr.sh|ocr_c256_mimo_7b_rl_ckpt100"
  "mimo_7b_rl|mimo_7b_rl|st_tt_clip005_1e_6_opencodereasoning_mimo7b_c1024|scripts/train_code/mimo_7b_rl/jsd005/length/opsd_st_tt_clip005_c1024_1e6_ocr.sh|ocr_c1024_mimo_7b_rl_ckpt100"
)

declare -a BASE_EVALS=(
  "deepseek_r1_1.5b|${MODEL_ROOT}/deepseek-r1-distill-qwen-1.5b|deepseek-r1-distill-qwen-1.5b"
  "falcon_h1r_7b|${MODEL_ROOT}/falcon-h1r-7b|falcon-h1r-7b"
  "mimo_7b_rl|${MODEL_ROOT}/mimo-7b-rl|mimo-7b-rl"
)

chmod +x \
  scripts/train_code/deepseek_r1_1.5b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh \
  scripts/train_code/deepseek_r1_1.5b/jsd005/length/*.sh \
  scripts/train_code/falcon_h1r_7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh \
  scripts/train_code/falcon_h1r_7b/jsd005/length/*.sh \
  scripts/train_code/mimo_7b_rl/jsd005/opsd_student_think_teacher_think_clip005_1e_6_opencodereasoning.sh \
  scripts/train_code/mimo_7b_rl/jsd005/length/*.sh \
  scripts/train_code/ocr/sbatch_watch.sh \
  scripts/train_code/ocr/watch_then_eval.sh \
  scripts/train_code/ocr/submit_three_code_evals.sh \
  scripts/eval_code/deepseek_r1_1.5b/*.sh \
  scripts/eval_code/falcon_h1r_7b/*.sh \
  scripts/eval_code/mimo_7b_rl/*.sh

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "[error] missing dataset ${DATASET_PATH}" >&2
  exit 1
fi
echo "[submit] dataset ready: ${DATASET_PATH}"

{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript\teval_tag"
} >"${MANIFEST_STAMPED}"

echo "[submit] root=${ROOT}"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r model_key model_tag run_name script eval_tag <<<"${entry}"
  if [[ ! -f "${script}" ]]; then
    echo "[error] missing ${script}" >&2
    exit 1
  fi
  jid=$(sbatch --parsable --chdir="${ROOT}" "${script}")
  echo -e "${model_key}\t${model_tag}\t${run_name}\t${jid}\t${script}\t${eval_tag}" >>"${MANIFEST_STAMPED}"
  echo "[submit] ${eval_tag} -> train_jid=${jid} script=${script}"
done

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --job-name=watch_ocr_r1fm \
  --output=log/train_code/ocr_r1_falcon_mimo/watch_slurm.%j.out \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}",CKPT_STEP=100 \
  "${ROOT}/scripts/train_code/ocr/sbatch_watch.sh")
echo "[submit] watch -> jid=${watch_jid}"

if [[ "${SUBMIT_BASE_EVAL}" == "1" ]]; then
  for be in "${BASE_EVALS[@]}"; do
    IFS='|' read -r model_key ckpt eval_tag <<<"${be}"
    echo "[submit] base evals MODEL_KEY=${model_key} ckpt=${ckpt} tag=${eval_tag}"
    MODEL_KEY="${model_key}" \
    CHECKPOINT_PATH="${ckpt}" \
    EVAL_TAG="${eval_tag}" \
    BASE_DIR="${ROOT}" \
    bash "${ROOT}/scripts/train_code/ocr/submit_three_code_evals.sh"
  done
fi

echo "[submit] done"
cat "${MANIFEST}"
