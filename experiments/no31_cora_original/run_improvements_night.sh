#!/usr/bin/env bash
# Overnight improvement pipeline after the CLEAN R4 batch finishes.
# 1) RR5 test eval  2) RR1 test-logit capture  3) post-hoc gate (CPU)
# 4) gate_tune6 training batch (4 trials)  5) final summaries.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
unset S2_NIGHT_EPOCHS
PY=/home/bbx/anaconda3/envs/minigpt4/bin/python
test -x "$PY"

NIGHT_LOG="$HERE/logs/NIGHT_IMPROVE_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$NIGHT_LOG") 2>&1
echo "[NIGHT] start $(date)"

# --- Wait until the CLEAN R4 batch summary exists (runner finished) ---
SUMMARY="$HERE/out/CORA_CLEAN_RANK4_MOE6_EARLY5_FIX1_20261111_summary.csv"
for i in $(seq 1 720); do
  if [ -f "$SUMMARY" ]; then break; fi
  sleep 60
done
if [ ! -f "$SUMMARY" ]; then
  echo "[NIGHT][FATAL] CLEAN R4 summary never appeared; aborting pipeline" >&2
  exit 2
fi
echo "[NIGHT] CLEAN R4 batch complete:"
cat "$SUMMARY"

# --- GPU should now be free; safety wait for memory release ---
sleep 30

# --- Phase A1: RR5 test evaluation ---
echo "========== [NIGHT] RR5 TEST EVAL =========="
"$PY" -u "$HERE/cora_rr5_best_test.py" --run \
  || echo "[NIGHT][WARN] RR5 test failed; continuing" >&2

# --- Phase A2: RR1 test logits capture ---
echo "========== [NIGHT] RR1 TEST LOGITS =========="
"$PY" -u "$HERE/cora_rr1_test_logits.py" --run \
  || echo "[NIGHT][WARN] RR1 test logits failed; continuing" >&2

# --- Phase B: post-hoc gate on CPU ---
echo "========== [NIGHT] POSTHOC GATE =========="
if [ -f "$HERE/out/CORA_RR1_TEST_LOGITS_20261112/layer_logits_test.npz" ]; then
  "$PY" "$ROOT/cora_posthoc_gate_test.py" \
    || echo "[NIGHT][WARN] posthoc gate failed; continuing" >&2
else
  echo "[NIGHT][WARN] test NPZ missing; skip posthoc" >&2
fi

# --- Phase C: gate tuning training batch ---
echo "========== [NIGHT] GATE TUNE TRAINING =========="
"$PY" -u "$HERE/cora_gate_tune6.py" --cpu-test
"$PY" -u "$HERE/cora_gate_tune6.py" --plan
"$PY" -u "$HERE/cora_gate_tune6.py" --run

# --- Final summaries ---
echo "========== [NIGHT] FINAL SUMMARY =========="
"$PY" -u "$HERE/cora_gate_tune6.py" --summary || true
echo "[NIGHT] all done $(date)"
