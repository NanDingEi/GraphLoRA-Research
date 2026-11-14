#!/usr/bin/env bash
# Watch the gate_tune training batch; exit when all four trials finish,
# the launcher dies, a traceback appears, or a log stalls >30 min.
EXP_DIR="/home/bbx/user_lx/CoRA_official/experiments/no31_cora_original"
LOG_DIR="$EXP_DIR/logs/CORA_GATE_TUNE_20261112"
STALL_LIMIT=1800

while true; do
  sleep 90
  done_count=0
  for name in G1_UGATE_B05 G2_UGATE_B00 G3_FIXED_A025 G4_ROUTE10_UGATE; do
    log="$LOG_DIR/$name.log"
    [ -f "$log" ] && grep -q "\[JOINT8 EARLY STOP\]\|Training time" "$log" && done_count=$((done_count+1))
  done
  if [ "$done_count" -eq 4 ]; then
    echo "MONITOR_STATUS=GATE_TUNE_ALL_DONE"
    exit 0
  fi
  if ! pgrep -f "cora_gate_tune6.py --run" >/dev/null && \
     ! pgrep -f "cora_gate_tune6.py --_worker" >/dev/null; then
    echo "MONITOR_STATUS=GATE_TUNE_DEAD done=$done_count/4"
    exit 1
  fi
  newest_log=$(ls -t "$LOG_DIR"/G*.log 2>/dev/null | head -1)
  if [ -n "$newest_log" ] && grep -q "Traceback" "$newest_log"; then
    echo "MONITOR_STATUS=TRACEBACK in $newest_log"
    grep -B2 -A10 "Traceback" "$newest_log" | tail -25
    exit 2
  fi
  if [ -n "$newest_log" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$newest_log") ))
    if [ "$age" -gt "$STALL_LIMIT" ]; then
      echo "MONITOR_STATUS=STALLED log=$newest_log age=${age}s"
      exit 3
    fi
  fi
done
