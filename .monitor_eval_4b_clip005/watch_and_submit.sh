#!/bin/bash
set -euo pipefail
BASE=/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis
STATE_DIR="$BASE/.monitor_eval_4b_clip005"
STATE_FILE="$STATE_DIR/submitted.txt"
touch "$STATE_FILE"
cd "$BASE"

# job_id|eval_tag|job_name_prefix|ckpt_root
RUNS=(
  "3216569|snt_tnt_clip005_acc70_4b_ckpt100|snt_tnt_clip005_acc70_4b|$BASE/outputs/qwen3_4b/snt_tnt_clip005_1e_6_ot_acc70_4b/3216569"
  "3216571|snt_tnt_clip005_acc0375_4b_ckpt100|snt_tnt_clip005_acc0375_4b|$BASE/outputs/qwen3_4b/snt_tnt_clip005_1e_6_ot_acc0375_4b/3216571"
  "3216572|snt_tnt_clip005_ota_4b_ckpt100|snt_tnt_clip005_ota_4b|$BASE/outputs/qwen3_4b/snt_tnt_clip005_1e_6_openthoughts_answer/3216572"
  "3216574|snt_tnt_clip005_ios_4b_ckpt100|snt_tnt_clip005_ios_4b|$BASE/outputs/qwen3_4b/snt_tnt_clip005_1e_6_openthoughts_irr_other_sol/3216574"
)

ckpt_ready() {
  local ckpt="$1/checkpoint-100"
  [[ -d "$ckpt" ]] || return 1
  [[ -f "$ckpt/config.json" ]] || return 1
  if [[ -f "$ckpt/model.safetensors.index.json" ]] || ls "$ckpt"/*.safetensors >/dev/null 2>&1; then
    if find "$ckpt" -name '*.tmp' -o -name '*.incomplete' 2>/dev/null | grep -q .; then
      return 1
    fi
    return 0
  fi
  return 1
}

progress_line() {
  local jid="$1" root="$2"
  local step="?"
  local log
  log=$(ls "$BASE"/log/train/4b/*."$jid".out 2>/dev/null | head -1 || true)
  if [[ -n "${log:-}" ]]; then
    step=$(rg -o '[0-9]+/100' "$log" 2>/dev/null | tail -1 || echo "?")
  fi
  local ckpts
  ckpts=$(ls -d "$root"/checkpoint-* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ',' | sed 's/,$//')
  echo "jid=$jid step=${step:-?} ckpts=${ckpts:-none}"
}

submit_evals() {
  local tag="$1" prefix="$2" ckpt="$3" jid="$4"
  local ds short ejid
  local submitted=()
  for ds in aime24 aime25 aime26 hmmt25; do
    short=${ds/aime/a}; short=${short/hmmt/h}
    ejid=$(CHECKPOINT_PATH="$ckpt" EVAL_TAG="$tag" BASE_DIR="$BASE" \
      sbatch --parsable --job-name="${prefix}_${short}_nt" \
      "scripts/eval/4b/${ds}_nothink.sh")
    submitted+=("${ds}:${ejid}")
    echo "SUBMITTED jid=$jid ds=$ds eval_job=$ejid tag=$tag"
  done
  echo "$jid" >> "$STATE_FILE"
  echo "AGENT_LOOP_WAKE_4bclip {\"prompt\":\"train job $jid checkpoint-100 ready; submitted evals ${submitted[*]}; continue monitoring remaining runs\",\"jid\":\"$jid\",\"evals\":\"${submitted[*]}\"}"
}

STATUS_LINES=()
NEW_SUBMITS=0
PENDING=0
for entry in "${RUNS[@]}"; do
  IFS='|' read -r jid tag prefix root <<< "$entry"
  STATUS_LINES+=("$(progress_line "$jid" "$root")")
  if grep -qx "$jid" "$STATE_FILE"; then
    continue
  fi
  PENDING=$((PENDING + 1))
  if ckpt_ready "$root"; then
    submit_evals "$tag" "$prefix" "$root/checkpoint-100" "$jid"
    NEW_SUBMITS=$((NEW_SUBMITS + 1))
    PENDING=$((PENDING - 1))
  fi
done

echo "STATUS $(date '+%F %T') new_submits=$NEW_SUBMITS pending=$PENDING"
printf '%s\n' "${STATUS_LINES[@]}"

done_n=$(wc -l < "$STATE_FILE" | tr -d ' ')
if [[ "$done_n" -ge 4 ]]; then
  echo "AGENT_LOOP_WAKE_4bclip {\"prompt\":\"all 4 train jobs have 16 nothink evals submitted; stop the monitor loop\",\"done\":true}"
fi
