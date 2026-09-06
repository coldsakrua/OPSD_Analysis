#!/bin/bash
# Resubmit robustness seed trains/evals as resource-homogeneous mega arrays.
# - seed eval: 1 array per eval seed (1024 / 65536), all models/variants × 4 datasets
# - seed train: 1 array per (seed, resource group): 1.7b(2gpu) and olmo(4gpu) separate
# Top-k trains stay as individual jobs; watch is restarted with updated manifest.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
OUT_DIR="${ROOT}/log/train/robustness"
mkdir -p "${OUT_DIR}" "${OUT_DIR}/array" log/eval/robustness/array
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${OUT_DIR}/compress_arrays_${STAMP}.log"
exec > >(tee -a "${LOG}") 2>&1

echo "[compress] root=${ROOT} stamp=${STAMP}"

ckpt_ok() {
  local p=$1
  [[ -f "${p}/config.json" ]] || return 1
  [[ -f "${p}/model.safetensors" ]] \
    || [[ -f "${p}/model.safetensors.index.json" ]] \
    || compgen -G "${p}/model-*.safetensors" >/dev/null
}

find_latest_ckpt100() {
  local root=$1
  [[ -d "${root}" ]] || return 1
  local d
  for d in $(ls -1d "${root}"/*/ 2>/dev/null | sort -V -r); do
    if ckpt_ok "${d}checkpoint-100"; then
      echo "${d}checkpoint-100"
      return 0
    fi
  done
  return 1
}

abs_ckpt() {
  local p=$1
  if [[ "${p}" = /* ]]; then echo "${p}"; else echo "${ROOT}/${p}"; fi
}

# ---------- cancel only this-conversation seed eval arrays + seed trains ----------
echo "[compress] cancelling pending seed eval arrays 3477506-3477525 (if present)"
mapfile -t EVAL_OLD < <(squeue -u 2501210611 -h -o '%i %j' | awk '
  $2 ~ /^seed(1024|65536)_(17|olmo)$/ { split($1,a,"_"); print a[1] }' | sort -u)
if ((${#EVAL_OLD[@]} > 0)); then
  echo "[compress] scancel eval arrays: ${EVAL_OLD[*]}"
  scancel "${EVAL_OLD[@]}"
fi

echo "[compress] cancelling pending seed train singles (name contains _s1024_ / _s65536_)"
mapfile -t TRAIN_OLD < <(squeue -u 2501210611 -t PENDING -h -o '%i %j' | awk '
  $2 ~ /_s1024_/ || $2 ~ /_s65536_/ { print $1 }')
if ((${#TRAIN_OLD[@]} > 0)); then
  echo "[compress] scancel seed trains: ${#TRAIN_OLD[@]} jobs"
  scancel "${TRAIN_OLD[@]}"
fi
sleep 2

# ---------- seed train mega arrays (same resources together) ----------
VARIANTS=(c256 c1024 answer ios first256 uni256 last256)
TRAIN_WORKER="${ROOT}/scripts/train/robustness/sbatch_seed_train_array.sh"
NEW_MANIFEST="${OUT_DIR}/submit_latest.tsv"
TMP_MANIFEST="${OUT_DIR}/submit_${STAMP}.tsv"
{
  echo -e "# model_key\tmodel_tag\trun_name\ttrain_jid\tscript"
} >"${TMP_MANIFEST}"

# keep topk rows from previous manifest if still in queue
if [[ -f "${NEW_MANIFEST}" ]]; then
  while IFS=$'\t' read -r mk mt rn jid sc; do
    [[ "${mk}" == \#* || -z "${mk}" ]] && continue
    if [[ "${rn}" == *seed1024* || "${rn}" == *seed65536* ]]; then
      continue
    fi
    # topk / other: keep if still queued/running
    if squeue -j "${jid}" -h -o '%i' 2>/dev/null | grep -q .; then
      echo -e "${mk}\t${mt}\t${rn}\t${jid}\t${sc}" >>"${TMP_MANIFEST}"
    else
      # still keep topk entries even if briefly missing? prefer keep if sacct pending/running
      st=$(sacct -j "${jid}" -n -X -o State -P 2>/dev/null | head -1 | tr -d ' ' || true)
      case "${st}" in
        PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED)
          echo -e "${mk}\t${mt}\t${rn}\t${jid}\t${sc}" >>"${TMP_MANIFEST}"
          ;;
        *)
          echo "[compress] drop finished/missing non-seed row ${rn} jid=${jid} state=${st:-?}"
          ;;
      esac
    fi
  done <"${NEW_MANIFEST}"
fi

submit_train_group() {
  local seed=$1 model_key=$2 model_tag=$3 suffix=$4 gpus=$5 cpus=$6 mem=$7 exclude=$8
  local man="${OUT_DIR}/train_tasks_seed${seed}_${suffix}_${STAMP}.tsv"
  : >"${man}"
  local v script run_name
  for v in "${VARIANTS[@]}"; do
    script="scripts/train/${model_tag}/jsd005/seed/opsd_st_tt_clip005_${v}_seed${seed}_ot.sh"
    # model_tag dir uses underscores matching scripts path
    if [[ "${model_key}" == "qwen3_1.7b" ]]; then
      script="scripts/train/qwen3_1.7b/jsd005/seed/opsd_st_tt_clip005_${v}_seed${seed}_ot.sh"
      run_name="st_tt_clip005_${v}_seed${seed}_1p7b"
    else
      script="scripts/train/olmo3_7b_think/jsd005/seed/opsd_st_tt_clip005_${v}_seed${seed}_ot.sh"
      run_name="st_tt_clip005_${v}_seed${seed}_olmo7bt"
    fi
    if [[ ! -f "${ROOT}/${script}" ]]; then
      echo "[error] missing ${script}" >&2
      continue
    fi
    echo -e "${model_key}\t${model_tag}\t${run_name}\t${script}" >>"${man}"
  done
  local n
  n=$(grep -cvE '^\s*(#|$)' "${man}" || true)
  if [[ "${n}" -eq 0 ]]; then
    echo "[error] empty train manifest ${man}" >&2
    return 1
  fi
  local last=$((n - 1))
  local jid
  jid=$(sbatch --parsable --chdir="${ROOT}" \
    --array="0-${last}" \
    --job-name="tr_s${seed}_${suffix}" \
    --cpus-per-task="${cpus}" \
    --gres="gpu:${gpus}" \
    --mem="${mem}" \
    --time=12:00:00 \
    --exclude="${exclude}" \
    --export=ALL,BASE_DIR="${ROOT}",TASK_MANIFEST="${man}" \
    "${TRAIN_WORKER}")
  jid="${jid%%;*}"
  echo "[compress] train array seed=${seed} group=${suffix} jid=${jid} tasks=0-${last} (gpus=${gpus})"
  local i=0
  while IFS=$'\t' read -r mk mt rn sc; do
    [[ -z "${mk}" || "${mk}" == \#* ]] && continue
    echo -e "${mk}\t${mt}\t${rn}\t${jid}_${i}\t${sc}" >>"${TMP_MANIFEST}"
    i=$((i + 1))
  done <"${man}"
}

for seed in 1024 65536; do
  submit_train_group "${seed}" "qwen3_1.7b" "qwen3_1.7b" "1p7b" 2 14 "220G" "gpua800n13"
  submit_train_group "${seed}" "olmo3_7b_think" "olmo3_7b_think" "olmo" 4 28 "400G" \
    "gpua800n03,gpua800n10,gpua800n13,gpua800n21"
done

cp -f "${TMP_MANIFEST}" "${NEW_MANIFEST}"
echo "[compress] updated manifest=${NEW_MANIFEST}"
wc -l "${NEW_MANIFEST}"

# ---------- seed eval mega arrays (1 per eval seed) ----------
declare -A RUNS_1P7=(
  [c256]=st_tt_clip005_1e_6_ot_1p7b_c256
  [c1024]=st_tt_clip005_1e_6_openthoughts_1p7b
  [answer]=st_tt_clip005_1e_6_openthoughts_answer_1p7b
  [ios]=st_tt_clip005_1e_6_openthoughts_irr_other_sol_1p7b
  [first256]=st_tt_clip005_c1024_first256_1p7b
  [uni256]=st_tt_clip005_c1024_uni256_1p7b
  [last256]=st_tt_clip005_c1024_last256_1p7b
)
declare -A RUNS_OLMO=(
  [c256]=st_tt_clip005_1e_6_ot_olmo7bt_c256
  [c1024]=st_tt_clip005_1e_6_openthoughts_olmo7bt
  [answer]=st_tt_clip005_1e_6_openthoughts_answer_olmo7bt
  [ios]=st_tt_clip005_1e_6_openthoughts_irr_other_sol_olmo7bt
  [first256]=st_tt_clip005_c1024_first256_olmo7bt
  [uni256]=st_tt_clip005_c1024_uni256_olmo7bt
  [last256]=st_tt_clip005_c1024_last256_olmo7bt
)
DATASETS=(aime24 aime25 aime26 hmmt25)
EVAL_WORKER="${ROOT}/scripts/eval/common/sbatch_seed_eval_mega_array.sh"

submit_eval_seed() {
  local seed=$1
  local man="${OUT_DIR}/eval_tasks_seed${seed}_${STAMP}.tsv"
  : >"${man}"
  local v run ckpt ds eval_tag out_json

  for v in "${VARIANTS[@]}"; do
    run="${RUNS_1P7[$v]}"
    if ckpt=$(find_latest_ckpt100 "${ROOT}/outputs/qwen3_1.7b/${run}"); then
      ckpt=$(abs_ckpt "${ckpt}")
      eval_tag="${run}_$(basename "${ckpt}")_seed${seed}"
      for ds in "${DATASETS[@]}"; do
        out_json="${ROOT}/eval_outputs/${eval_tag}/${ds}_1.7b_think_seed${seed}.json"
        echo -e "qwen3_1.7b\t${ROOT}/scripts/eval/1.7b\t_think\t${ckpt}\t${eval_tag}\t${seed}\t${ds}\t${out_json}" >>"${man}"
      done
    else
      echo "[compress][miss] 1.7b ${v}"
    fi
  done

  for v in "${VARIANTS[@]}"; do
    run="${RUNS_OLMO[$v]}"
    if ckpt=$(find_latest_ckpt100 "${ROOT}/outputs/olmo3_7b_think/${run}"); then
      ckpt=$(abs_ckpt "${ckpt}")
      eval_tag="${run}_$(basename "${ckpt}")_seed${seed}"
      for ds in "${DATASETS[@]}"; do
        out_json="${ROOT}/eval_outputs/${eval_tag}/${ds}_olmo_7b_think_sgl_seed${seed}.json"
        echo -e "olmo3_7b_think\t${ROOT}/scripts/eval/olmo_7b_think\t_sgl\t${ckpt}\t${eval_tag}\t${seed}\t${ds}\t${out_json}" >>"${man}"
      done
    else
      echo "[compress][miss] olmo ${v}"
    fi
  done

  local n last jid
  n=$(grep -cvE '^\s*(#|$)' "${man}" || true)
  if [[ "${n}" -eq 0 ]]; then
    echo "[compress] no eval tasks for seed=${seed}"
    return 0
  fi
  last=$((n - 1))
  jid=$(sbatch --parsable --chdir="${ROOT}" \
    --array="0-${last}" \
    --job-name="ev_s${seed}" \
    --export=ALL,BASE_DIR="${ROOT}",TASK_MANIFEST="${man}" \
    "${EVAL_WORKER}")
  jid="${jid%%;*}"
  echo "[compress] eval array seed=${seed} jid=${jid} tasks=0-${last} (n=${n})"
  echo "${jid}" >"${OUT_DIR}/eval_array_seed${seed}_${STAMP}.txt"
}

for seed in 1024 65536; do
  submit_eval_seed "${seed}"
done

# ---------- restart watch on new manifest ----------
OLD_WATCH=$(cat "${OUT_DIR}/watch_jid_20260906_032824.txt" 2>/dev/null || true)
# also find running robust_watch_eval
mapfile -t WATCHES < <(squeue -u 2501210611 -h -o '%i %j' | awk '$2=="robust_watch_eval"{print $1}')
if ((${#WATCHES[@]} > 0)); then
  echo "[compress] scancel old watch: ${WATCHES[*]}"
  scancel "${WATCHES[@]}"
fi
# clear stale done stamps for rescinded seed trains (optional: only seed-related)
# keep topk done stamps if any
watch_jid=$(sbatch --parsable --chdir="${ROOT}" \
  --export=ALL,BASE_DIR="${ROOT}",MANIFEST="${NEW_MANIFEST}" \
  "${ROOT}/scripts/train/robustness/sbatch_watch.sh")
watch_jid="${watch_jid%%;*}"
echo "[compress] new watch -> ${watch_jid}"
echo "${watch_jid}" >"${OUT_DIR}/watch_jid_${STAMP}.txt"

echo "[compress] queue snapshot (robustness-related):"
squeue -u 2501210611 -o '%.18i %.12P %.28j %.2t %.10M' | awk '
  NR==1 || /tr_s|ev_s|robust_watch|topk|snt_tnt_clip005|st_tt_clip005_topk/ {print}'
echo "[compress] total queue lines: $(squeue -u 2501210611 -h | wc -l)"
echo "[compress] log=${LOG}"
