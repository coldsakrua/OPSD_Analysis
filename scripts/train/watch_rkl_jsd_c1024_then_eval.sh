#!/bin/bash
# Watch rkl/jsd c1024 train jobs; as each finishes with checkpoint-100,
# submit the four-dataset eval suite for that model.
#
# Usage (login node, from OPSD_Analysis or anywhere):
#   bash scripts/train/watch_rkl_jsd_c1024_then_eval.sh
#   MANIFEST=log/train/rkl_jsd_c1024_submit_latest.tsv INTERVAL=120 \
#     bash scripts/train/watch_rkl_jsd_c1024_then_eval.sh
set -euo pipefail

BASE=/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis
MANIFEST=${MANIFEST:-${BASE}/log/train/rkl_jsd_c1024_submit_latest.tsv}
INTERVAL=${INTERVAL:-120}
POST_DONE_WAIT=${POST_DONE_WAIT:-60}
LOG_DIR=${BASE}/log/train
LOG="${LOG_DIR}/watch_rkl_jsd_c1024_then_eval.$(date +%Y%m%d_%H%M%S).log"
STAMP_DIR="${LOG_DIR}/watch_rkl_jsd_c1024_stamps"
mkdir -p "${LOG_DIR}" "${STAMP_DIR}"
cd "${BASE}"

exec > >(tee -a "${LOG}") 2>&1

DATASETS=(aime24 aime25 aime26 hmmt25)

log() { echo "[watch $(date '+%F %T')] $*"; }

short_ds() {
  case "$1" in
    aime24) echo a24 ;;
    aime25) echo a25 ;;
    aime26) echo a26 ;;
    hmmt25) echo h25 ;;
    *) echo "$1" ;;
  esac
}

ckpt_ready() {
  local ckpt="$1"
  [[ -f "${ckpt}/config.json" ]] || return 1
  if [[ -f "${ckpt}/model.safetensors" \
     || -f "${ckpt}/model-00001-of-00001.safetensors" \
     || -f "${ckpt}/model.safetensors.index.json" ]]; then
    return 0
  fi
  return 1
}

resolve_ckpt() {
  local out_dir="$1"
  if ckpt_ready "${out_dir}/checkpoint-100"; then
    echo "${out_dir}/checkpoint-100"
    return 0
  fi
  if ckpt_ready "${out_dir}/final"; then
    echo "${out_dir}/final"
    return 0
  fi
  return 1
}

# model_tag -> eval script dir + suffix
eval_script_for() {
  local model_tag="$1" ds="$2"
  case "${model_tag}" in
    qwen3_1.7b)
      echo "scripts/eval/1.7b/${ds}_think.sh"
      ;;
    qwen3_4b)
      echo "scripts/eval/4b/${ds}_think.sh"
      ;;
    qwen3_4b_thinking)
      echo "scripts/eval/qwen3_4b_thinking/${ds}_think.sh"
      ;;
    olmo3_7b_think)
      echo "scripts/eval/olmo_7b_think/${ds}_sgl.sh"
      ;;
    olmo3_7b_instruct)
      echo "scripts/eval/olmo3_7b_instruct/${ds}_sgl.sh"
      ;;
    *)
      echo ""
      ;;
  esac
}

eval_log_subdir() {
  local model_tag="$1" ds="$2"
  case "${model_tag}" in
    qwen3_1.7b) echo "log/eval/1.7b/${ds}/think" ;;
    qwen3_4b) echo "log/eval/4b/${ds}/think" ;;
    qwen3_4b_thinking) echo "log/eval/qwen3_4b_thinking/${ds}/think" ;;
    olmo3_7b_think) echo "log/eval/olmo_7b_think/${ds}/sgl" ;;
    olmo3_7b_instruct) echo "log/eval/olmo3_7b_instruct/${ds}/sgl" ;;
    *) echo "log/eval/${model_tag}/${ds}" ;;
  esac
}

job_in_queue() {
  local jid="$1"
  local st
  st=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
  [[ -n "${st}" ]]
}

job_state() {
  local jid="$1"
  squeue -j "${jid}" -h -o '%T %R' 2>/dev/null || echo 'GONE'
}

submit_four_for() {
  local jid="$1" model_tag="$2" loss="$3" run_name="$4"
  local out_dir ckpt eval_tag script ds job_name log_sub jid_eval
  out_dir="${BASE}/outputs/${model_tag}/${run_name}/${jid}"
  if ! ckpt=$(resolve_ckpt "${out_dir}"); then
    log "[WARN] job=${jid} left queue but no ready ckpt under ${out_dir}"
    return 1
  fi
  eval_tag="${run_name}_ckpt100"
  log "submitting 4 evals: model=${model_tag} loss=${loss} job=${jid}"
  log "  ckpt=${ckpt}"
  log "  eval_tag=${eval_tag}"

  for ds in "${DATASETS[@]}"; do
    script=$(eval_script_for "${model_tag}" "${ds}")
    if [[ -z "${script}" || ! -f "${script}" ]]; then
      log "[ERROR] missing eval script for ${model_tag}/${ds}: ${script}"
      continue
    fi
    log_sub=$(eval_log_subdir "${model_tag}" "${ds}")
    mkdir -p "${log_sub}"
    job_name="${eval_tag}_$(short_ds "${ds}")"
    # keep job-name short for squeue display
    if ((${#job_name} > 48)); then
      job_name="${loss}_${model_tag##*_}_ckpt100_$(short_ds "${ds}")"
    fi
    jid_eval=$(
      sbatch --parsable \
        --job-name="${job_name}" \
        --output="${log_sub}/%x.%j.out" \
        --export=ALL,BASE_DIR="${BASE}",CHECKPOINT_PATH="${ckpt}",EVAL_TAG="${eval_tag}" \
        "${script}"
    )
    log "  ${ds} -> ${jid_eval} (${script})"
    sleep 1
  done
  echo "${ckpt}" > "${STAMP_DIR}/${jid}.submitted"
  return 0
}

if [[ ! -f "${MANIFEST}" ]]; then
  log "[ERROR] manifest missing: ${MANIFEST}"
  exit 1
fi

# Load jobs from TSV (skip header)
declare -a JOB_IDS=()
declare -A MODEL_TAG=()
declare -A LOSS=()
declare -A RUN_NAME=()

while IFS=$'\t' read -r jid model_tag loss run_name _script; do
  [[ "${jid}" == "job_id" ]] && continue
  [[ -z "${jid}" ]] && continue
  JOB_IDS+=("${jid}")
  MODEL_TAG["${jid}"]="${model_tag}"
  LOSS["${jid}"]="${loss}"
  RUN_NAME["${jid}"]="${run_name}"
done < "${MANIFEST}"

log "start log=${LOG}"
log "manifest=${MANIFEST} jobs=${#JOB_IDS[@]} interval=${INTERVAL}s"
for jid in "${JOB_IDS[@]}"; do
  log "  track ${jid} ${MODEL_TAG[$jid]} ${LOSS[$jid]} ${RUN_NAME[$jid]}"
done

declare -A DONE=()
for jid in "${JOB_IDS[@]}"; do
  if [[ -f "${STAMP_DIR}/${jid}.submitted" ]]; then
    DONE["${jid}"]=1
    log "already submitted evals for ${jid} ($(cat "${STAMP_DIR}/${jid}.submitted"))"
  fi
done

while true; do
  pending=0
  echo "---- $(date '+%F %T') ----"
  for jid in "${JOB_IDS[@]}"; do
    mt="${MODEL_TAG[$jid]}"
    loss="${LOSS[$jid]}"
    rn="${RUN_NAME[$jid]}"
    out="${BASE}/outputs/${mt}/${rn}/${jid}"
    st=$(job_state "${jid}")
    has_ckpt=no
    if resolve_ckpt "${out}" >/dev/null 2>&1; then
      has_ckpt=yes
    fi
    flag=""
    [[ -n "${DONE[$jid]:-}" ]] && flag=" EVAL_SUBMITTED"
    log "job=${jid} ${mt}/${loss} state=${st} ckpt100=${has_ckpt}${flag}"

    if [[ -n "${DONE[$jid]:-}" ]]; then
      continue
    fi

    if job_in_queue "${jid}"; then
      pending=$((pending + 1))
      continue
    fi

    # left queue — wait briefly for filesystem flush then submit
    log "job ${jid} left queue; wait ${POST_DONE_WAIT}s then check ckpt"
    sleep "${POST_DONE_WAIT}"
    if submit_four_for "${jid}" "${mt}" "${loss}" "${rn}"; then
      DONE["${jid}"]=1
    else
      # keep retrying next loop (late write)
      pending=$((pending + 1))
      log "[WARN] will retry ${jid} next round"
    fi
  done

  remaining=0
  for jid in "${JOB_IDS[@]}"; do
    [[ -z "${DONE[$jid]:-}" ]] && remaining=$((remaining + 1))
  done
  log "remaining=${remaining} (queue_or_retry=${pending})"
  if [[ "${remaining}" -eq 0 ]]; then
    log "all ${#JOB_IDS[@]} train jobs have evals submitted"
    log "done"
    exit 0
  fi
  sleep "${INTERVAL}"
done
