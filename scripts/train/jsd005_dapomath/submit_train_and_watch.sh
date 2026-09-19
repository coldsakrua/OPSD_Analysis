#!/bin/bash
# Submit DAPO-Math jsd005 st_tt / snt_tt trains for Qwen3-1.7B and Qwen3-4B,
# write manifest, then start a CPU-node watcher that submits ckpt-100 think evals.
#
# If a train parquet is missing, the matching CPU preprocess job is submitted
# first and the train waits on it (--dependency=afterok).
#
# Usage (from OPSD_Analysis):
#   bash scripts/train/jsd005_dapomath/submit_train_and_watch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"

OUT_DIR="${ROOT}/log/train/jsd005_dapomath"
mkdir -p "${OUT_DIR}" log/train/{1.7b,4b} log/data
MANIFEST="${OUT_DIR}/submit_latest.tsv"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_STAMPED="${OUT_DIR}/submit_${STAMP}.tsv"

# model_key|model_tag|run_name|train_script|eval_tag|dataset_parquet|preprocess_script
declare -a JOBS=(
  "qwen3_1.7b|qwen3_1.7b|st_tt_clip005_1e_6_dapomath_1p7b|scripts/train/qwen3_1.7b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_dapomath.sh|st_tt_dapo_1p7b_ckpt100|data/dapo/preprocessed/dapo-math-17k.opsd.solution.sthink_tthink.qwen3_1.7b.maxprompt1024.parquet|scripts/data/preprocess_opsd_dapo_qwen3_1.7b_st_tt.sh"
  "qwen3_4b|qwen3_4b|st_tt_clip005_1e_6_dapomath_4b|scripts/train/qwen3_4b/jsd005/opsd_student_think_teacher_think_clip005_1e_6_dapomath.sh|st_tt_dapo_4b_ckpt100|data/dapo/preprocessed/dapo-math-17k.opsd.solution.sthink_tthink.qwen3_4b.maxprompt1024.parquet|scripts/data/preprocess_opsd_dapo_qwen3_4b_st_tt.sh"
  "qwen3_1.7b|qwen3_1.7b|snt_tt_clip005_1e_6_dapomath_1p7b|scripts/train/qwen3_1.7b/jsd005/opsd_student_nothink_teacher_think_clip005_1e_6_dapomath.sh|snt_tt_dapo_1p7b_ckpt100|data/dapo/preprocessed/dapo-math-17k.opsd.solution.snothink_tthink.qwen3_1.7b.maxprompt1024.parquet|scripts/data/preprocess_opsd_dapo_qwen3_1.7b_snt_tt.sh"
  "qwen3_4b|qwen3_4b|snt_tt_clip005_1e_6_dapomath_4b|scripts/train/qwen3_4b/jsd005/opsd_student_nothink_teacher_think_clip005_1e_6_dapomath.sh|snt_tt_dapo_4b_ckpt100|data/dapo/preprocessed/dapo-math-17k.opsd.solution.snothink_tthink.qwen3_4b.maxprompt1024.parquet|scripts/data/preprocess_opsd_dapo_qwen3_4b_snt_tt.sh"
)

{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript\teval_tag"
} >"${MANIFEST_STAMPED}"

echo "[submit] root=${ROOT}"
for entry in "${JOBS[@]}"; do
  IFS='|' read -r model_key model_tag run_name script eval_tag dataset preprocess <<<"${entry}"
  if [[ ! -f "${script}" ]]; then
    echo "[error] missing ${script}" >&2
    exit 1
  fi
  chmod +x "${script}" "${preprocess}"

  extra_sbatch=()
  if [[ ! -f "${dataset}" ]]; then
    echo "[submit] missing ${dataset}; submitting preprocess ${preprocess}"
    pp_jid=$(sbatch --parsable --chdir="${ROOT}" "${preprocess}")
    pp_jid="${pp_jid%%;*}"
    echo "[submit] preprocess ${eval_tag} -> jid=${pp_jid}"
    extra_sbatch+=(--dependency="afterok:${pp_jid}")
  fi

  jid=$(sbatch --parsable --chdir="${ROOT}" "${extra_sbatch[@]}" "${script}")
  jid="${jid%%;*}"
  echo -e "${model_key}\t${model_tag}\t${run_name}\t${jid}\t${script}\t${eval_tag}" >>"${MANIFEST_STAMPED}"
  echo "[submit] ${eval_tag} -> jid=${jid} script=${script}"
done

cp -f "${MANIFEST_STAMPED}" "${MANIFEST}"
echo "[submit] manifest=${MANIFEST}"
echo "[submit] stamped=${MANIFEST_STAMPED}"

chmod +x "${ROOT}/scripts/train/jsd005_dapomath/sbatch_watch.sh" \
  "${ROOT}/scripts/train/jsd005_dapomath/watch_then_eval.sh" \
  "${ROOT}/scripts/train/jsd005_dapomath/submit_four_think_evals.sh"

watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${MANIFEST}" \
  "${ROOT}/scripts/train/jsd005_dapomath/sbatch_watch.sh")
watch_jid="${watch_jid%%;*}"
echo "[submit] watch -> jid=${watch_jid}"
echo "[submit] done"
cat "${MANIFEST}"
