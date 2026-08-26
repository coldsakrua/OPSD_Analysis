#!/bin/bash
# Launch snt_tnt length train→nothink-eval monitor for 1.7b/4b/8b.
set -euo pipefail

BASE_DIR=${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "${BASE_DIR}"
mkdir -p log/monitor

MONITOR_LOG="log/monitor/watch_snt_tnt_length_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts/monitor/watch_snt_tnt_length_train_eval_nothink.sh > "${MONITOR_LOG}" 2>&1 &
MPID=$!
echo "${MPID}" > log/monitor/watch_snt_tnt_length.pid

echo "monitor_pid=${MPID}"
echo "monitor_log=${MONITOR_LOG}"
echo "tail -f ${MONITOR_LOG}"
