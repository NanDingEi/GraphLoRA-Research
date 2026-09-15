#!/bin/bash
# usage: bash gpu_monitor.sh <tag>   -> logs to logs_monitor/gpu_<tag>.log
TAG=${1:-run}
mkdir -p logs_monitor
while true; do
  TS=$(date '+%m-%d %H:%M:%S')
  LINE=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null)
  PROCS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')
  echo "$TS | $LINE | procs: $PROCS" >> logs_monitor/gpu_${TAG}.log
  sleep 60
done
