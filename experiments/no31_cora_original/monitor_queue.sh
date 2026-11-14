#!/usr/bin/env bash
# Queue-phase monitor: wake when the queue's final summary tag (HP8) is DONE,
# the driver dies, a traceback appears in any active HP log, or a log stalls.
EXP="/home/bbx/user_lx/CoRA_official/experiments/no31_cora_original"
STATUS="$EXP/logs/night_queue_status.log"
STALL=1800
while true; do
  sleep 180
  if grep -qE "DONE G10" "$STATUS" 2>/dev/null; then
    echo "MONITOR_STATUS=QUEUE_ALL_DONE"
    tail -14 "$STATUS"
    cat "$EXP/out/CORA_HIER_FULLSTACK_20261113/hier_fullstack_valid_summary.csv" 2>/dev/null
    exit 0
  fi
  if ! pgrep -f "night_queue_driver.sh" >/dev/null; then
    echo "MONITOR_STATUS=DRIVER_DEAD"; tail -6 "$STATUS"; exit 1
  fi
  for log in "$EXP"/logs/CORA_HIER_FULLSTACK_20261113/HP*.log \
             "$EXP"/logs/CORA_GATE_TUNE_20261112/G[67]_*.log; do
    [ -f "$log" ] || continue
    if grep -q "Traceback" "$log"; then
      echo "MONITOR_STATUS=TRACEBACK in $log"
      grep -A6 "Traceback" "$log" | tail -12
      exit 2
    fi
    # Skip finished runs; only live logs can stall.
    if grep -qE "\[JOINT8 EARLY STOP\]|Training time" "$log"; then continue; fi
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if [ "$age" -gt "$STALL" ]; then
      echo "MONITOR_STATUS=STALLED $log age=${age}s"; exit 3
    fi
  done
done
