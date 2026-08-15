#!/bin/bash
set -euo pipefail
BASE=/gpfs/share/home/2501210611/opsd_analysis/OPSD_Analysis
cd "$BASE"
mkdir -p log/train/{4b,1.7b}

# Cancel stuck PD jobs from prior over-aggressive exclude (ReqNodeNotAvail n25)
for jid in 3226360 3226362 3226369; do
  scancel "$jid" 2>/dev/null || true
  echo "scancel $jid"
done
sleep 1

# Occupied now: n01,n10,n21,n24 + always n13; also avoid known-bad n25
EXCL=gpua800n01,gpua800n10,gpua800n13,gpua800n21,gpua800n24,gpua800n25
echo "exclude=$EXCL"

# 4bi c256/c512 already RUNNING (3226358/3226359) — skip
SCRIPTS=(
  "scripts/train/qwen3_4b/jsd005/length/opsd_st_tt_clip005_c256_1e6_ot.sh"
  "scripts/train/qwen3_4b/jsd005/length/opsd_st_tt_clip005_c512_1e6_ot.sh"
  "scripts/train/qwen3_1.7b/jsd005/length/opsd_st_tt_clip005_c256_1e6_ot.sh"
  "scripts/train/qwen3_1.7b/jsd005/length/opsd_st_tt_clip005_c512_1e6_ot.sh"
)

for s in "${SCRIPTS[@]}"; do
  jid=$(BASE_DIR="$BASE" sbatch --parsable --exclude="$EXCL" "$s")
  echo "SUBMITTED $jid  $s"
  sleep 0.5
done

echo '==== QUEUE ===='
squeue -u 2501210611 -o '%.18i %.9P %.28j %.2t %.10M %N %R'
