#!/usr/bin/env bash
# Overnight pipeline v3: RR5 test-logit capture (8 min) -> gate_tune training.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
unset S2_NIGHT_EPOCHS
PY=/home/bbx/anaconda3/envs/minigpt4/bin/python
test -x "$PY"

NIGHT_LOG="$HERE/logs/NIGHT_IMPROVE_V3_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$NIGHT_LOG") 2>&1
echo "[NIGHT-V3] start $(date)"

for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -lt 2000 ]; then break; fi
  sleep 10
done

echo "========== [NIGHT-V3] RR5 TEST LOGITS =========="
if [ ! -f "$HERE/out/CORA_RR5_TEST_LOGITS_20261112/layer_logits_test.npz" ]; then
  "$PY" -u "$HERE/cora_rr5_test_logits.py" --run \
    || echo "[NIGHT-V3][WARN] RR5 test logits failed; continuing" >&2
else
  echo "[NIGHT-V3] RR5 NPZ already present; skip capture"
fi

echo "========== [NIGHT-V3] GATE TUNE TRAINING =========="
"$PY" -u "$HERE/cora_gate_tune6.py" --run

echo "========== [NIGHT-V3] FINAL SUMMARY =========="
"$PY" -u "$HERE/cora_gate_tune6.py" --summary || true
echo "[NIGHT-V3] all done $(date)"
