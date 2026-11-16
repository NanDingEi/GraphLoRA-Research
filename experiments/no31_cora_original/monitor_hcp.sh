#!/usr/bin/env bash
# HCP night monitor: exit when both H runs finish (summary CSV), a traceback
# appears, the training process dies, or a log stalls >25 min.
EXP_DIR="/home/bbx/user_lx/CoRA_official/experiments/no31_cora_original"
SUMMARY="$EXP_DIR/out/CORA_HIER_CLUSTER_MOE_20261112_summary.csv"
NIGHT_LOG="$EXP_DIR/logs/HCP_NIGHT_20261112.log"
STALL=1500
while true; do
  sleep 90
  if [ -f "$SUMMARY" ]; then echo "MONITOR_STATUS=HCP_ALL_DONE"; cat "$SUMMARY"; exit 0; fi
  if grep -q "Traceback" "$NIGHT_LOG" 2>/dev/null; then
    echo "MONITOR_STATUS=TRACEBACK"; grep -A8 "Traceback" "$NIGHT_LOG" | tail -15; exit 2
  fi
  if ! pgrep -f "cora_hier_cluster_moe6" >/dev/null; then
    echo "MONITOR_STATUS=HCP_PROCESS_DEAD"; exit 1
  fi
  log=$(ls -t "$EXP_DIR"/logs/CORA_HIER_CLUSTER_MOE_20261112_*/HIER_MOE.log 2>/dev/null | head -1)
  if [ -n "$log" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if [ "$age" -gt "$STALL" ]; then echo "MONITOR_STATUS=STALLED $log age=${age}s"; exit 3; fi
  fi
done
