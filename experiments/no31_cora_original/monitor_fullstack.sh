#!/usr/bin/env bash
# Fullstack-phase monitor: wake when HP1/HP2 summary appears, the chain
# dies, or a traceback/stall shows up.
EXP="/home/bbx/user_lx/CoRA_official/experiments/no31_cora_original"
SUMMARY="$EXP/out/CORA_HIER_FULLSTACK_20261113/hier_fullstack_valid_summary.csv"
FS_LOG="$EXP/logs/HIER_FULLSTACK_NIGHT.log"
STALL=1800
while true; do
  sleep 120
  if [ -f "$SUMMARY" ]; then echo "MONITOR_STATUS=FULLSTACK_HP12_DONE"; cat "$SUMMARY"; exit 0; fi
  if grep -q "Traceback" "$FS_LOG" 2>/dev/null; then
    echo "MONITOR_STATUS=FULLSTACK_TRACEBACK"; grep -A8 "Traceback" "$FS_LOG" | tail -15; exit 2
  fi
  if ! pgrep -f "run_hier_fullstack.sh" >/dev/null && \
     ! pgrep -f "cora_hier_fullstack.py" >/dev/null && \
     ! pgrep -f "night_queue_driver.sh" >/dev/null; then
    echo "MONITOR_STATUS=FULLSTACK_AND_DRIVER_DEAD"; exit 1
  fi
  log=$(ls -t "$EXP"/logs/CORA_HIER_FULLSTACK_20261113/HP*.log 2>/dev/null | head -1)
  if [ -n "$log" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if [ "$age" -gt "$STALL" ]; then echo "MONITOR_STATUS=FULLSTACK_STALLED $log age=${age}s"; exit 3; fi
  fi
done
