#!/usr/bin/env bash
# Queue monitor: fire when G14 (final cleanup) is DONE, driver dies,
# traceback, or a live log stalls.
EXP="/home/bbx/user_lx/CoRA_official/experiments/no31_cora_original"
STATUS="$EXP/logs/night_queue_status.log"
STALL=1800
echo "[MONITOR] armed $(date +%F_%T); watching DONE P8 in $STATUS"
while true; do
  sleep 180
  LAST_G=$(grep -oE '^[G][0-9]+' "$EXP/night_queue.txt" 2>/dev/null | tail -1)
  if [ -n "$LAST_G" ] && grep -qE "DONE $LAST_G" "$STATUS" 2>/dev/null; then
    echo "MONITOR_STATUS=QUEUE_ALL_DONE"
    tail -8 "$STATUS"
    cat "$EXP/out/CORA_HIER_FULLSTACK_20261113/hier_fullstack_valid_summary.csv" 2>/dev/null
    exit 0
  fi
  if ! pgrep -f "night_queue_driver.sh" >/dev/null; then
    echo "MONITOR_STATUS=DRIVER_DEAD"; tail -4 "$STATUS"; exit 1
  fi
  for log in "$EXP"/logs/CORA_HIER_FULLSTACK_20261113/HP*.log \
             "$EXP"/logs/CORA_GATE_TUNE_20261112/G*.log; do
    [ -f "$log" ] || continue
    grep -q "Traceback" "$log" && { echo "MONITOR_STATUS=TRACEBACK in $log"; exit 2; }
    grep -qE "\[JOINT8 EARLY STOP\]|Training time" "$log" && continue
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if [ "$age" -gt "$STALL" ]; then
      echo "MONITOR_STATUS=STALLED $log age=${age}s"; exit 3
    fi
  done
done
