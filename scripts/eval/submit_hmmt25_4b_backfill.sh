#!/bin/bash
# Backfill HMMT25 evals for qwen3-4b (+ think/nothink) and qwen3-4b-instruct (nothink only).
#
# Coverage:
#   1) 4b OT think/nothink combos: snt_tnt / snt_tt / st_tnt  → think + nothink
#   2) 4b student + instruct teacher (s4b_tit): sol / nogt / same → think + nothink
#   3) instruct different prompts: base / answer / enc / irr / enc_tr / irr_tr → nothink
#   4) instruct hyper sweeps → nothink
#
# Usage (from repo root):
#   bash scripts/eval/submit_hmmt25_4b_backfill.sh
#   DRY_RUN=1 bash scripts/eval/submit_hmmt25_4b_backfill.sh
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OUT=${BASE_DIR}/outputs
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}

mkdir -p "${BASE_DIR}/log/eval" "${BASE_DIR}/log/monitor"
JOBS_FILE="${BASE_DIR}/log/monitor/hmmt25_backfill_$(date +%Y%m%d_%H%M%S).txt"
: > "${JOBS_FILE}"

log() { echo "[$(date '+%F %T')] $*"; }

ckpt_ready() {
  local ckpt="$1"
  [[ -d "${ckpt}" ]] || return 1
  [[ -f "${ckpt}/config.json" ]] || return 1
  [[ -f "${ckpt}/model.safetensors.index.json" || -f "${ckpt}/model.safetensors" || -f "${ckpt}/pytorch_model.bin" ]] || return 1
  return 0
}

already_done() {
  local tag="$1" script="$2"
  local base stem out
  base=$(basename "${script}" .sh)   # hmmt25_nothink / hmmt25_think
  case "${script}" in
    */4b_instruct/*) stem="hmmt25_4b_instruct_${base#hmmt25_}" ;;
    */4b/*)          stem="hmmt25_4b_${base#hmmt25_}" ;;
    *) return 1 ;;
  esac
  out="${BASE_DIR}/eval_outputs/${tag}/${stem}.json"
  # also accept .metrics.json as done signal
  [[ -f "${out}" || -f "${BASE_DIR}/eval_outputs/${tag}/${stem}.metrics.json" ]]
}

submit_one() {
  local ckpt="$1" tag="$2" prefix="$3" script="$4"
  local mode ds job_name job_id

  if ! ckpt_ready "${ckpt}"; then
    log "SKIP missing ckpt: ${ckpt}"
    return 0
  fi
  if [[ "${SKIP_EXISTING}" == "1" ]] && already_done "${tag}" "${script}"; then
    log "SKIP existing: ${tag} $(basename "${script}")"
    return 0
  fi

  case "$(basename "${script}")" in
    *_nothink.sh) mode=nt ;;
    *_think.sh)   mode=th ;;
    *)            mode=eval ;;
  esac
  ds=$(basename "${script}" | sed -E 's/_(nothink|think)\.sh$//')
  job_name="${prefix}_${ds/hmmt25/h25}_${mode}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY sbatch ${job_name} tag=${tag} script=$(basename "${script}")"
    echo "DRY ${job_name} ${tag} ${script}" >> "${JOBS_FILE}"
    return 0
  fi

  cd "${BASE_DIR}"
  job_id=$(
    sbatch --parsable \
      --job-name="${job_name}" \
      --export=ALL,BASE_DIR="${BASE_DIR}",CHECKPOINT_PATH="${ckpt}",EVAL_TAG="${tag}" \
      "${script}" 2>/dev/null | tail -1
  )
  log "submitted ${job_id} ${job_name} tag=${tag}"
  echo "${job_id} ${job_name} ${tag}" >> "${JOBS_FILE}"
}

E4B_NT="${BASE_DIR}/scripts/eval/4b/hmmt25_nothink.sh"
E4B_TH="${BASE_DIR}/scripts/eval/4b/hmmt25_think.sh"
E4BI_NT="${BASE_DIR}/scripts/eval/4b_instruct/hmmt25_nothink.sh"

# --- 1) qwen3-4b OT combos (no st_tt checkpoint) ---
submit_one "${OUT}/qwen3_4b/snt_tnt_1e_6_openthoughts/2960119/checkpoint-100" \
  snt_tnt_1e6_ot_ckpt100 snt_tnt_ot "${E4B_NT}"
submit_one "${OUT}/qwen3_4b/snt_tnt_1e_6_openthoughts/2960119/checkpoint-100" \
  snt_tnt_1e6_ot_ckpt100 snt_tnt_ot "${E4B_TH}"

submit_one "${OUT}/qwen3_4b/snt_tt_1e_6_openthoughts/2947311/checkpoint-100" \
  snt_tt_1e6_ot_ckpt100 snt_tt_ot "${E4B_NT}"
submit_one "${OUT}/qwen3_4b/snt_tt_1e_6_openthoughts/2947311/checkpoint-100" \
  snt_tt_1e6_ot_ckpt100 snt_tt_ot "${E4B_TH}"

submit_one "${OUT}/qwen3_4b/st_tnt_1e_6_openthoughts/2960125/checkpoint-100" \
  st_tnt_1e6_ot_ckpt100 st_tnt_ot "${E4B_NT}"
submit_one "${OUT}/qwen3_4b/st_tnt_1e_6_openthoughts/2960125/checkpoint-100" \
  st_tnt_1e6_ot_ckpt100 st_tnt_ot "${E4B_TH}"

# --- 2) qwen3-4b student + instruct teacher ---
submit_one "${OUT}/qwen3_4b/s4b_tit_opsd_sol_1e_6/2963103/checkpoint-100" \
  s4b_tit_opsd_sol_1e6_ckpt100 s4b_tit_sol "${E4B_NT}"
submit_one "${OUT}/qwen3_4b/s4b_tit_opsd_sol_1e_6/2963103/checkpoint-100" \
  s4b_tit_opsd_sol_1e6_ckpt100 s4b_tit_sol "${E4B_TH}"

submit_one "${OUT}/qwen3_4b/s4b_tit_opsd_nogt_1e_6/2963105/checkpoint-100" \
  s4b_tit_opsd_nogt_1e6_ckpt100 s4b_tit_nogt "${E4B_NT}"
submit_one "${OUT}/qwen3_4b/s4b_tit_opsd_nogt_1e_6/2963105/checkpoint-100" \
  s4b_tit_opsd_nogt_1e6_ckpt100 s4b_tit_nogt "${E4B_TH}"

submit_one "${OUT}/qwen3_4b/s4b_tit_opsd_same_1e_6/2963682/checkpoint-100" \
  s4b_tit_opsd_same_1e6_ckpt100 s4b_tit_same "${E4B_NT}"
submit_one "${OUT}/qwen3_4b/s4b_tit_opsd_same_1e_6/2963682/checkpoint-100" \
  s4b_tit_opsd_same_1e6_ckpt100 s4b_tit_same "${E4B_TH}"

# --- 3) instruct different prompts (nothink only) ---
submit_one "${OUT}/qwen3_4b_instruct/snt_tnt_1e_6_openthoughts_instruct/2960131/checkpoint-100" \
  snt_tnt_oti_ckpt100 snt_tnt_oti "${E4BI_NT}"
submit_one "${OUT}/qwen3_4b_instruct/snt_tnt_1e_6_openthoughts_answer_instruct/2960160/checkpoint-100" \
  snt_tnt_otai_ckpt100 snt_tnt_otai "${E4BI_NT}"
submit_one "${OUT}/qwen3_4b_instruct/snt_tnt_encourage_ot_1e_6_instruct/2960161/checkpoint-100" \
  snt_tnt_enc_oti_ckpt100 snt_tnt_enc_oti "${E4BI_NT}"
submit_one "${OUT}/qwen3_4b_instruct/snt_tnt_irrelevant_ot_1e_6_instruct/2960162/checkpoint-100" \
  snt_tnt_irr_oti_ckpt100 snt_tnt_irr_oti "${E4BI_NT}"
submit_one "${OUT}/qwen3_4b_instruct/snt_tnt_encourage_trans_ot_1e_6_instruct/2962408/checkpoint-100" \
  snt_tnt_enc_tr_oti_ckpt100 snt_tnt_enc_tr_oti "${E4BI_NT}"
submit_one "${OUT}/qwen3_4b_instruct/snt_tnt_irrelevant_trans_ot_1e_6_instruct/2962409/checkpoint-100" \
  snt_tnt_irr_tr_oti_ckpt100 snt_tnt_irr_tr_oti "${E4BI_NT}"

# --- 4) instruct hyper (nothink only) ---
submit_one "${OUT}/qwen3_4b_instruct/snt_tnt_lr2e6_oti/2965228/checkpoint-100" \
  snt_oti_lr2e6_ckpt100 snt_oti_lr2e6 "${E4BI_NT}"
submit_one "${OUT}/qwen3_4b_instruct/snt_tnt_lr5e6_oti/2965229/checkpoint-100" \
  snt_oti_lr5e6_ckpt100 snt_oti_lr5e6 "${E4BI_NT}"
submit_one "${OUT}/qwen3_4b_instruct/snt_tnt_jsd1e7_oti/2965230/checkpoint-100" \
  snt_oti_c1e7_ckpt100 snt_oti_c1e7 "${E4BI_NT}"
submit_one "${OUT}/qwen3_4b_instruct/snt_tnt_jsd1e5_oti/2965231/checkpoint-100" \
  snt_oti_c1e5_ckpt100 snt_oti_c1e5 "${E4BI_NT}"
submit_one "${OUT}/qwen3_4b_instruct/snt_tnt_gbs16_oti/2965232/checkpoint-100" \
  snt_oti_g16_ckpt100 snt_oti_g16 "${E4BI_NT}"
submit_one "${OUT}/qwen3_4b_instruct/snt_tnt_gbs64_oti/2965233/checkpoint-100" \
  snt_oti_g64_ckpt100 snt_oti_g64 "${E4BI_NT}"
submit_one "${OUT}/qwen3_4b_instruct/snt_tnt_gbs64_jsd1e7_oti/2966749/checkpoint-100" \
  snt_oti_g64c1e7_ckpt100 snt_oti_g64c1e7 "${E4BI_NT}"
submit_one "${OUT}/qwen3_4b_instruct/snt_tnt_lr5e6_jsd1e5_oti/2966719/checkpoint-100" \
  snt_oti_l5c1e5_ckpt100 snt_oti_l5c1e5 "${E4BI_NT}"

log "done. jobs file: ${JOBS_FILE}"
wc -l "${JOBS_FILE}"
