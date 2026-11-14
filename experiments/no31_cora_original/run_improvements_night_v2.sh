#!/usr/bin/env bash
# Overnight improvement pipeline v2: fixed RR1 logits install, then gate_tune.
# Runs immediately: logits -> posthoc -> gate_tune(G1..G4).
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
unset S2_NIGHT_EPOCHS
PY=/home/bbx/anaconda3/envs/minigpt4/bin/python
test -x "$PY"

NIGHT_LOG="$HERE/logs/NIGHT_IMPROVE_V2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$NIGHT_LOG") 2>&1
echo "[NIGHT-V2] start $(date)"

# --- Wait for GPU to be free of the previous gate_tune worker ---
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -lt 2000 ]; then break; fi
  sleep 10
done

# --- Phase A2 (fixed): RR1 test logits capture ---
echo "========== [NIGHT-V2] RR1 TEST LOGITS (fixed) =========="
if [ ! -f "$HERE/out/CORA_RR1_TEST_LOGITS_20261112/layer_logits_test.npz" ]; then
  "$PY" -u "$HERE/cora_rr1_test_logits.py" --run \
    || echo "[NIGHT-V2][WARN] RR1 test logits failed; continuing" >&2
else
  echo "[NIGHT-V2] test NPZ already present; skip capture"
fi

# --- Phase B: post-hoc gate on CPU ---
echo "========== [NIGHT-V2] POSTHOC GATE =========="
if [ -f "$HERE/out/CORA_RR1_TEST_LOGITS_20261112/layer_logits_test.npz" ]; then
  "$PY" "$ROOT/cora_posthoc_gate_test.py" \
    || echo "[NIGHT-V2][WARN] posthoc gate failed; continuing" >&2
else
  echo "[NIGHT-V2][WARN] test NPZ missing; skip posthoc" >&2
fi

# --- Phase C: gate tuning training batch ---
echo "========== [NIGHT-V2] GATE TUNE TRAINING =========="
"$PY" -u "$HERE/cora_gate_tune6.py" --cpu-test
"$PY" -u "$HERE/cora_gate_tune6.py" --plan
"$PY" -u "$HERE/cora_gate_tune6.py" --run

echo "========== [NIGHT-V2] FINAL SUMMARY =========="
"$PY" -u "$HERE/cora_gate_tune6.py" --summary || true
echo "[NIGHT-V2] all done $(date)"
