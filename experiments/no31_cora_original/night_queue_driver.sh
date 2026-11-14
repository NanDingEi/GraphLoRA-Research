#!/usr/bin/env bash
# Indefinite night-queue driver: consumes TAG|command lines from the queue
# file serially, forever. Waits while the chained fullstack pipeline holds
# the GPU; also starts consuming if that pipeline dies without a summary.
# Lines are tagged; a tag already marked DONE/FAIL in the status file is
# skipped, so re-running the driver or crash-restarts are safe.
EXP="/home/bbx/user_lx/CoRA_official/experiments/no31_cora_original"
ROOT="/home/bbx/user_lx/CoRA_official"
QUEUE="$EXP/night_queue.txt"
STATUS="$EXP/logs/night_queue_status.log"
DRIVER_LOG="$EXP/logs/night_queue_driver.log"
FS_SUMMARY="$EXP/out/CORA_HIER_FULLSTACK_20261113/hier_fullstack_valid_summary.csv"
PY=/home/bbx/anaconda3/envs/minigpt4/bin/python
touch "$QUEUE" "$STATUS"
cd "$ROOT"

fs_busy() {
  pgrep -f "run_hier_fullstack.sh" >/dev/null || \
  pgrep -f "cora_hier_fullstack.py" >/dev/null
}

while true; do
  # Disk guard: checkpoint saves need ~0.5G each; wait when low.
  free_mb=$(df -Pm / | awk 'NR==2 {print $4}')
  if [ "$free_mb" -lt 6000 ]; then
    echo "[$(date +%F_%T)] WARN disk free=${free_mb}MB < 6000MB; pausing queue" >> "$STATUS"
    sleep 600
    continue
  fi
  if [ ! -f "$FS_SUMMARY" ] && fs_busy; then sleep 120; continue; fi
  if [ ! -f "$FS_SUMMARY" ]; then
    # Chained pipeline died without summary; give recovery 10 minutes.
    if [ $((SECONDS % 600)) -lt 2 ]; then
      echo "[$(date +%F_%T)] WARN fullstack chain dead without summary; consuming queue anyway" >> "$STATUS"
    fi
  fi
  next=""
  while IFS= read -r line; do
    case "$line" in \#*|"") continue ;; esac
    tag="${line%%|*}"
    if ! grep -qE "(DONE|FAIL) $tag( |$)" "$STATUS" 2>/dev/null; then
      next="$line"; break
    fi
  done < "$QUEUE"
  if [ -z "$next" ]; then sleep 180; continue; fi
  tag="${next%%|*}"
  cmd="${next#*|}"
  echo "[$(date +%F_%T)] START $tag :: $cmd" >> "$STATUS"
  if bash -c "$cmd" >> "$DRIVER_LOG" 2>&1; then
    echo "[$(date +%F_%T)] DONE $tag" >> "$STATUS"
  else
    echo "[$(date +%F_%T)] FAIL $tag" >> "$STATUS"
  fi
  sleep 20
done
