#!/bin/bash
set -euo pipefail
BASE=/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis
STATE_DIR="$BASE/.monitor_eval_longgen6"
STATE_FILE="$STATE_DIR/submitted.txt"
touch "$STATE_FILE"
cd "$BASE"

# job_id|family|mode|eval_tag|job_prefix|ckpt_root
# family: 1p7b | 4bi ; mode: think | nothink
RUNS=(
  "3219795|1p7b|think|st_tt_clip005_1p7b_c2048_ckpt100|st_tt_clip005_c2048_1p7b|$BASE/outputs/qwen3_1.7b/st_tt_clip005_1e_6_ot_1p7b_c2048/3219795"
  "3219796|1p7b|think|st_tt_clip005_1p7b_c4096_ckpt100|st_tt_clip005_c4096_1p7b|$BASE/outputs/qwen3_1.7b/st_tt_clip005_1e_6_ot_1p7b_c4096/3219796"
  "3219797|1p7b|think|st_tt_clip005_1p7b_c6144_ckpt100|st_tt_clip005_c6144_1p7b|$BASE/outputs/qwen3_1.7b/st_tt_clip005_1e_6_ot_1p7b_c6144/3219797"
  "3219872|4bi|nothink|snt_tnt_clip005_oti_c2048_ckpt100|snt_tnt_clip005_c2048_4bi|$BASE/outputs/qwen3_4b_instruct/snt_tnt_clip005_1e_6_oti_c2048/3219872"
  "3219873|4bi|nothink|snt_tnt_clip005_oti_c4096_ckpt100|snt_tnt_clip005_c4096_4bi|$BASE/outputs/qwen3_4b_instruct/snt_tnt_clip005_1e_6_oti_c4096/3219873"
  "3219874|4bi|nothink|snt_tnt_clip005_oti_c6144_ckpt100|snt_tnt_clip005_c6144_4bi|$BASE/outputs/qwen3_4b_instruct/snt_tnt_clip005_1e_6_oti_c6144/3219874"
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
  local step="?" state="?"
  local log
  log=$(ls "$BASE"/log/train/*/*."$jid".out 2>/dev/null | head -1 || true)
  if [[ -n "${log:-}" ]]; then
    step=$(rg -o '[0-9]+/100' "$log" 2>/dev/null | tail -1 || echo "?")
  fi
  state=$(squeue -j "$jid" -h -o '%T' 2>/dev/null || echo "GONE")
  local ckpts
  ckpts=$(ls -d "$root"/checkpoint-* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ',' | sed 's/,$//' || true)
  echo "jid=$jid state=${state:-?} step=${step:-?} ckpts=${ckpts:-none}"
}

current_exclude() {
  local nodes
  nodes=$(squeue -u 2501210611 -h -t R -o '%N' | tr ',' '\n' | sort -u | paste -sd, -)
  if [[ -n "$nodes" ]]; then
    echo "${nodes},gpua800n13"
  else
    echo "gpua800n13"
  fi
}

submit_evals() {
  local family="$1" mode="$2" tag="$3" prefix="$4" ckpt="$5" jid="$6"
  local ds short ejid script suffix excl
  local submitted=()
  for ds in aime24 aime25 aime26 hmmt25; do
    short=${ds/aime/a}; short=${short/hmmt/h}
    if [[ "$mode" == "think" ]]; then
      suffix=th
    else
      suffix=nt
    fi
    if [[ "$family" == "1p7b" ]]; then
      script="scripts/eval/1.7b/${ds}_${mode}.sh"
    else
      script="scripts/eval/4b_instruct/${ds}_${mode}.sh"
    fi
    excl=$(current_exclude)
    ejid=$(CHECKPOINT_PATH="$ckpt" EVAL_TAG="$tag" BASE_DIR="$BASE" \
      sbatch --parsable --exclude="$excl" --job-name="${prefix}_${short}_${suffix}" "$script")
    submitted+=("${ds}:${ejid}")
    echo "SUBMITTED jid=$jid ds=$ds mode=$mode eval_job=$ejid tag=$tag exclude=$excl"
    sleep 1
  done
  echo "$jid" >> "$STATE_FILE"
  echo "AGENT_LOOP_WAKE_longgen6 {\"prompt\":\"train job $jid checkpoint-100 ready; submitted ${mode} evals ${submitted[*]}; continue monitoring remaining runs\",\"jid\":\"$jid\",\"evals\":\"${submitted[*]}\"}"
}

STATUS_LINES=()
NEW_SUBMITS=0
PENDING=0
for entry in "${RUNS[@]}"; do
  IFS='|' read -r jid family mode tag prefix root <<< "$entry"
  STATUS_LINES+=("$(progress_line "$jid" "$root")")
  if grep -qx "$jid" "$STATE_FILE"; then
    continue
  fi
  PENDING=$((PENDING + 1))
  if ckpt_ready "$root"; then
    submit_evals "$family" "$mode" "$tag" "$prefix" "$root/checkpoint-100" "$jid"
    NEW_SUBMITS=$((NEW_SUBMITS + 1))
    PENDING=$((PENDING - 1))
  fi
done

echo "STATUS $(date '+%F %T') new_submits=$NEW_SUBMITS pending=$PENDING"
printf '%s\n' "${STATUS_LINES[@]}"

done_n=$(sort -u "$STATE_FILE" | wc -l | tr -d ' ')
if [[ "$done_n" -ge 6 ]]; then
  echo "AGENT_LOOP_WAKE_longgen6 {\"prompt\":\"all 6 longgen train jobs have evals submitted; stop the monitor loop\",\"done\":true}"
fi
