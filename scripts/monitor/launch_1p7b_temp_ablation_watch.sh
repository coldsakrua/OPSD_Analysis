#!/bin/bash
# Launch 1.7b temperature-ablation train watcher for known train job IDs.
set -euo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${BASE_DIR}"
mkdir -p log/monitor

J1=${1:?job1}
J2=${2:?job2}
J3=${3:?job3}

MONITOR_LOG="log/monitor/watch_1p7b_temp_ablation_$(date +%Y%m%d_%H%M%S).log"
export SPECS_OVERRIDE="${J1}|st_tt_clip005_s06_t06_1e6_openthoughts_1p7b|st_tt_clip005_s06_t06_1e6_1p7b_ckpt100|st_tt_s06t06 ${J2}|st_tt_clip005_s06_t11_1e6_openthoughts_1p7b|st_tt_clip005_s06_t11_1e6_1p7b_ckpt100|st_tt_s06t11 ${J3}|st_tt_clip005_s11_t06_1e6_openthoughts_1p7b|st_tt_clip005_s11_t06_1e6_1p7b_ckpt100|st_tt_s11t06"

nohup bash scripts/monitor/watch_1p7b_temp_ablation_train_eval.sh > "${MONITOR_LOG}" 2>&1 &
MPID=$!
echo "${MPID}" > "log/monitor/watch_1p7b_temp_ablation.pid"
echo "MONITOR_PID=${MPID}"
echo "MONITOR_LOG=${MONITOR_LOG}"
sleep 2
head -n 40 "${MONITOR_LOG}"
