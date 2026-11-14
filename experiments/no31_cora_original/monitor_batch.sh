#!/usr/bin/env bash
# Monitor the 6-run CLEAN R4 batch. Exit (thus notifying the agent) when:
#  - all 6 runs complete (summary CSV written), or
#  - the runner dies / a Traceback appears / a log stalls >25 min.
RUNNER_PID="$1"
EXP_DIR="/home/bbx/user_lx/CoRA_official/experiments/no31_cora_original"
SUMMARY="$EXP_DIR/out/CORA_CLEAN_RANK4_MOE6_EARLY5_FIX1_20261111_summary.csv"
PREFIX="CORA_CLEAN_R4_EARLY5_FIX1_20261111"
STALL_LIMIT=1500   # seconds without log growth

while true; do
  sleep 60
  if [ -f "$SUMMARY" ]; then
    echo "MONITOR_STATUS=ALL_DONE"
    cat "$SUMMARY"
    exit 0
  fi
  if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
    echo "MONITOR_STATUS=RUNNER_DEAD (summary CSV missing => crashed before finishing)"
    exit 1
  fi
  newest_log=$(ls -t "$EXP_DIR/logs/$PREFIX"*/CLEAN_R4_MOE.log 2>/dev/null | head -1)
  if [ -n "$newest_log" ]; then
    if grep -q "Traceback" "$newest_log"; then
      echo "MONITOR_STATUS=TRACEBACK in $newest_log"
      grep -A5 "Traceback" "$newest_log" | tail -20
      exit 2
    fi
    age=$(( $(date +%s) - $(stat -c %Y "$newest_log") ))
    if [ "$age" -gt "$STALL_LIMIT" ]; then
      echo "MONITOR_STATUS=STALLED log=$newest_log age=${age}s"
      exit 3
    fi
  fi
done
