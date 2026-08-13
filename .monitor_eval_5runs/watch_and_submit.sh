#!/bin/bash
set -euo pipefail
BASE=/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis
STATE_DIR="$BASE/.monitor_eval_5runs"
STATE_FILE="$STATE_DIR/submitted.txt"
touch "$STATE_FILE"
cd "$BASE"

# job_id|family|eval_tag|ckpt_root
# family: 4bi | olmo
RUNS=(
  "3213971|4bi|snt_tnt_jsd010_oti_ckpt100|$BASE/outputs/qwen3_4b_instruct/snt_tnt_jsd010_oti/3213971"
  "3213970|4bi|snt_tnt_jsd001_oti_ckpt100|$BASE/outputs/qwen3_4b_instruct/snt_tnt_jsd001_oti/3213970"
  "3213946|4bi|snt_tnt_clip005_1e6_oti_ckpt100|$BASE/outputs/qwen3_4b_instruct/snt_tnt_clip005_1e_6_openthoughts_instruct/3213946"
  "3213989|olmo|snt_tnt_clip005_1e6_ios_olmo_ckpt100|$BASE/outputs/olmo3_7b_instruct/snt_tnt_clip005_1e_6_openthoughts_irr_other_sol_olmo7bit/3213989"
  "3213949|olmo|snt_tnt_clip005_1e6_olmo7bi_ckpt100|$BASE/outputs/olmo3_7b_instruct/snt_tnt_clip005_1e_6_openthoughts_olmo7bit/3213949"
)

ckpt_ready() {
  local ckpt="$1/checkpoint-100"
  [[ -d "$ckpt" ]] || return 1
  # full HF save: config + weights index or single shard
  [[ -f "$ckpt/config.json" ]] || return 1
  if [[ -f "$ckpt/model.safetensors.index.json" ]] || [[ -f "$ckpt/model.safetensors" ]] || ls "$ckpt"/*.safetensors >/dev/null 2>&1; then
    # wait until trainer finished writing (no .tmp / incomplete)
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
  log=$(ls "$BASE"/log/train/*/*."$jid".out 2>/dev/null | head -1 || true)
  if [[ -n "${log:-}" ]]; then
    step=$(rg -o '[0-9]+/100' "$log" 2>/dev/null | tail -1 || echo "?")
  fi
  local ckpts
  ckpts=$(ls -d "$root"/checkpoint-* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ',' | sed 's/,$//')
  echo "jid=$jid step=${step:-?} ckpts=${ckpts:-none}"
}

submit_evals() {
  local family="$1" tag="$2" ckpt="$3" jid="$4"
  local ds short ejid script jobname
  local submitted=()
  for ds in aime24 aime25 aime26 hmmt25; do
    short=${ds/aime/a}; short=${short/hmmt/h}
    if [[ "$family" == "4bi" ]]; then
      script="scripts/eval/4b_instruct/${ds}_nothink.sh"
      jobname="e4bi_${tag%%_ckpt*}_${short}"
      # keep job-name short
      jobname="e_${jid}_${short}"
    else
      script="scripts/eval/olmo3_7b_instruct/${ds}_sgl.sh"
      jobname="e_${jid}_${short}"
    fi
    ejid=$(CHECKPOINT_PATH="$ckpt" EVAL_TAG="$tag" BASE_DIR="$BASE" \
      sbatch --parsable --job-name="$jobname" "$script")
    submitted+=("${ds}:${ejid}")
    echo "SUBMITTED jid=$jid ds=$ds eval_job=$ejid tag=$tag ckpt=$ckpt"
  done
  echo "$jid" >> "$STATE_FILE"
  echo "AGENT_LOOP_WAKE_eval5 {\"prompt\":\"train job $jid checkpoint-100 ready; submitted evals ${submitted[*]}; continue monitoring remaining runs\",\"jid\":\"$jid\",\"evals\":\"${submitted[*]}\"}"
}

STATUS_LINES=()
NEW_SUBMITS=0
ALL_DONE=1
for entry in "${RUNS[@]}"; do
  IFS='|' read -r jid family tag root <<< "$entry"
  STATUS_LINES+=("$(progress_line "$jid" "$root")")
  if grep -qx "$jid" "$STATE_FILE"; then
    continue
  fi
  ALL_DONE=0
  if ckpt_ready "$root"; then
    submit_evals "$family" "$tag" "$root/checkpoint-100" "$jid"
    NEW_SUBMITS=$((NEW_SUBMITS + 1))
  fi
done

echo "STATUS $(date '+%F %T') new_submits=$NEW_SUBMITS"
printf '%s\n' "${STATUS_LINES[@]}"

if [[ "$ALL_DONE" -eq 1 ]] && [[ "$NEW_SUBMITS" -eq 0 ]]; then
  # all five already submitted previously
  n=$(wc -l < "$STATE_FILE" | tr -d ' ')
  if [[ "$n" -ge 5 ]]; then
    echo "AGENT_LOOP_WAKE_eval5 {\"prompt\":\"all 5 train jobs have evals submitted; stop the monitor loop\",\"done\":true}"
  fi
fi
