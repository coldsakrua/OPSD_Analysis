#!/bin/bash
# Watch 4 OPSD qwen3.5-4b train jobs; when all done, submit ckpt-100 evals.
set -euo pipefail
BASE=/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis
LOG="$BASE/log/train/qwen3.5_4b/watch_train_then_eval.$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

# job_id -> short_tag:output_subdir
declare -A JOBS=(
  [2986790]="snt_tt:snt_tt_1e_6_openthoughts_q35_4b"
  [2986870]="st_tt:st_tt_1e_6_openthoughts_q35_4b"
  [2986975]="snt_tnt:snt_tnt_1e_6_openthoughts_q35_4b"
  [2986983]="st_tnt:st_tnt_1e_6_openthoughts_q35_4b"
)

EVAL_SCRIPTS=(
  scripts/eval/qwen3.5_4b/aime24_nothink.sh
  scripts/eval/qwen3.5_4b/aime24_think.sh
  scripts/eval/qwen3.5_4b/aime25_nothink.sh
  scripts/eval/qwen3.5_4b/aime25_think.sh
  scripts/eval/qwen3.5_4b/aime26_nothink.sh
  scripts/eval/qwen3.5_4b/aime26_think.sh
  scripts/eval/qwen3.5_4b/hmmt25_nothink.sh
  scripts/eval/qwen3.5_4b/hmmt25_think.sh
)

echo "[watch] start $(date) log=$LOG"
echo "[watch] tracking jobs: ${!JOBS[*]}"

all_done() {
  local j
  for j in "${!JOBS[@]}"; do
    # still in queue?
    if squeue -j "$j" -h >/dev/null 2>&1; then
      local st
      st=$(squeue -j "$j" -h -o '%T' 2>/dev/null || true)
      if [[ -n "$st" ]]; then
        return 1
      fi
    fi
  done
  return 0
}

status_report() {
  echo "---- $(date) ----"
  squeue -u 2501210611 -o '%.10i %.12P %.24j %.2t %.10M %R' || true
  local j meta tag subdir out ckpt
  for j in 2986790 2986870 2986975 2986983; do
    meta="${JOBS[$j]}"
    tag="${meta%%:*}"
    subdir="${meta##*:}"
    out="$BASE/outputs/qwen3.5_4b/${subdir}/${j}"
    ckpt="$out/checkpoint-100"
    st=$(squeue -j "$j" -h -o '%T %R' 2>/dev/null || echo 'GONE')
    step=$(rg -o '[0-9]+%/100' "$BASE/log/train/qwen3.5_4b/"*"${j}.out" 2>/dev/null | tail -1 || true)
    echo "[watch] job=$j tag=$tag state=$st progress=${step:-?} ckpt100=$([ -d "$ckpt" ] && echo yes || echo no)"
  done
}

submit_evals() {
  cd "$BASE"
  local j meta tag subdir ckpt eval_tag script jid
  local submitted=0
  for j in 2986790 2986870 2986975 2986983; do
    meta="${JOBS[$j]}"
    tag="${meta%%:*}"
    subdir="${meta##*:}"
    ckpt="$BASE/outputs/qwen3.5_4b/${subdir}/${j}/checkpoint-100"
    if [[ ! -d "$ckpt" ]]; then
      # fallback to final/
      if [[ -d "$BASE/outputs/qwen3.5_4b/${subdir}/${j}/final" ]]; then
        ckpt="$BASE/outputs/qwen3.5_4b/${subdir}/${j}/final"
      else
        echo "[watch][ERROR] missing checkpoint for $tag job=$j under $BASE/outputs/qwen3.5_4b/${subdir}/${j}"
        continue
      fi
    fi
    eval_tag="${tag}_1e6_ot_q35_4b_ckpt100"
    echo "[watch] submitting 8 evals for $tag ckpt=$ckpt eval_tag=$eval_tag"
    for script in "${EVAL_SCRIPTS[@]}"; do
      jid=$(
        sbatch --parsable \
          --export=ALL,CHECKPOINT_PATH="${ckpt}",EVAL_TAG="${eval_tag}" \
          "$script"
      )
      echo "[watch]   $script -> $jid"
      submitted=$((submitted + 1))
      sleep 1
    done
  done
  echo "[watch] submitted $submitted eval jobs"
  squeue -u 2501210611 -o '%.10i %.12P %.24j %.2t %R' | head -50
}

# poll every 10 minutes
while true; do
  status_report
  if all_done; then
    echo "[watch] all 4 train jobs left the queue"
    # verify none failed without ckpt-100
    missing=0
    for j in 2986790 2986870 2986975 2986983; do
      meta="${JOBS[$j]}"
      subdir="${meta##*:}"
      if [[ ! -d "$BASE/outputs/qwen3.5_4b/${subdir}/${j}/checkpoint-100" && ! -d "$BASE/outputs/qwen3.5_4b/${subdir}/${j}/final" ]]; then
        echo "[watch][WARN] job $j finished but no checkpoint-100/final"
        missing=1
      fi
    done
    if [[ "$missing" -eq 1 ]]; then
      echo "[watch] waiting 10min more in case late writes..."
      sleep 600
      status_report
    fi
    submit_evals
    echo "[watch] done $(date)"
    exit 0
  fi
  echo "[watch] sleep 10min..."
  sleep 600
done
