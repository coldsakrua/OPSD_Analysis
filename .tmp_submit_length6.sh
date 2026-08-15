#!/bin/bash
set -euo pipefail
BASE=/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis
cd "$BASE"
mkdir -p log/train/{4b,4b-instruct,1.7b}

SCRIPTS=(
  "scripts/train/qwen3_4b_instruct/jsd005/length/opsd_snt_tnt_clip005_c256_1e6_ot.sh"
  "scripts/train/qwen3_4b_instruct/jsd005/length/opsd_snt_tnt_clip005_c512_1e6_ot.sh"
  "scripts/train/qwen3_4b/jsd005/length/opsd_st_tt_clip005_c256_1e6_ot.sh"
  "scripts/train/qwen3_4b/jsd005/length/opsd_st_tt_clip005_c512_1e6_ot.sh"
  "scripts/train/qwen3_1.7b/jsd005/length/opsd_st_tt_clip005_c256_1e6_ot.sh"
  "scripts/train/qwen3_1.7b/jsd005/length/opsd_st_tt_clip005_c512_1e6_ot.sh"
)

EXTRA_EXCL=()

current_exclude() {
  local nodes
  nodes=$(squeue -u 2501210611 -h -t R -o '%N' 2>/dev/null | tr ',' '\n' | grep -oE 'gpua800n[0-9]+' | sort -u || true)
  printf '%s\n' $nodes "${EXTRA_EXCL[@]}" gpua800n13 | awk 'NF' | sort -u | paste -sd, -
}

wait_for_node() {
  local jid="$1" max_sec="${2:-240}" elapsed=0 st node
  while (( elapsed < max_sec )); do
    st=$(squeue -j "$jid" -h -o '%T' 2>/dev/null || echo GONE)
    node=$(squeue -j "$jid" -h -o '%N' 2>/dev/null || true)
    if [[ "$st" == "RUNNING" && -n "${node:-}" && "$node" != "(null)" ]]; then
      echo "$node"
      return 0
    fi
    case "$st" in
      GONE|FAILED|CANCELLED|NODE_FAIL|TIMEOUT|OUT_OF_MEMORY)
        echo ""
        return 1
        ;;
    esac
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo ""
  return 1
}

RESULTS=()
for s in "${SCRIPTS[@]}"; do
  excl=$(current_exclude)
  echo "---- submitting $s"
  echo "exclude=$excl"
  jid=$(BASE_DIR="$BASE" sbatch --parsable --exclude="$excl" "$s")
  echo "SUBMITTED jid=$jid"
  node=$(wait_for_node "$jid" 240 || true)
  if [[ -n "${node:-}" ]]; then
    echo "STARTED jid=$jid node=$node"
    while IFS= read -r n; do
      [[ -n "$n" ]] && EXTRA_EXCL+=("$n")
    done < <(echo "$node" | tr ',' '\n' | grep -oE 'gpua800n[0-9]+')
    RESULTS+=("$jid|$s|$node|R")
  else
    st=$(squeue -j "$jid" -h -o '%T %R' 2>/dev/null || echo GONE)
    echo "NOT_YET_RUNNING jid=$jid status=$st"
    RESULTS+=("$jid|$s|?|PD")
    sleep 2
  fi
done

echo '==== SUMMARY ===='
printf '%s\n' "${RESULTS[@]}"
echo '==== QUEUE ===='
squeue -u 2501210611 -o '%.18i %.9P %.28j %.2t %.10M %N %R'
