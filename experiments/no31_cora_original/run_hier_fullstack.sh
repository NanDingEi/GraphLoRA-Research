#!/usr/bin/env bash
# Full-stack hierarchical-cluster + personal routing: two lambda variants.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
unset S2_NIGHT_EPOCHS
PY=/home/bbx/anaconda3/envs/minigpt4/bin/python

SCRIPT="experiments/no31_cora_original/cora_hier_fullstack.py"
test -f "$SCRIPT"
"$PY" "$SCRIPT" --cpu-test

# Wait for the clean HCP batch to finish before taking the GPU.
CLEAN_SUMMARY="$HERE/out/CORA_HIER_CLUSTER_MOE_20261112_summary.csv"
for i in $(seq 1 480); do
  if [ -f "$CLEAN_SUMMARY" ]; then break; fi
  if ! pgrep -f "cora_hier_cluster_moe6" >/dev/null; then
    sleep 30
    if [ -f "$CLEAN_SUMMARY" ]; then break; fi
    echo "[HIER-FULL][FATAL] clean HCP batch dead without summary; aborting" >&2
    exit 2
  fi
  sleep 60
done
if [ ! -f "$CLEAN_SUMMARY" ]; then
  echo "[HIER-FULL][FATAL] clean summary never appeared" >&2
  exit 2
fi
echo "[HIER-FULL] clean HCP batch complete:"
cat "$CLEAN_SUMMARY"
sleep 30

for row in "HP1_HIER_L10 1.0" "HP2_HIER_L02 0.2"; do
  read -r name lambda <<< "$row"
  echo "========== [HIER-FULL START] ${name} lambda=${lambda} =========="
  "$PY" -u "$SCRIPT" --run --only "$name" \
    --protocol CORA_HIER_FULLSTACK_20261113 \
    --patience 10
  echo "========== [HIER-FULL DONE] ${name} =========="
done

"$PY" -u "$SCRIPT" --summary --protocol CORA_HIER_FULLSTACK_20261113
