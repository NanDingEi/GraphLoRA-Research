#!/usr/bin/env bash
# D1 controlled contrast on ONE GPU. Foreground only, one run at a time.
set -euo pipefail
cd /home/bbx/user_lx/GraphLoRA
MODE="${1:-both}"
GPU="${GPU:-0}"
case "$MODE" in
  A|B|both) ;;
  *) echo "Usage: GPU=0 bash run_D1.sh [A|B|both]" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"
S2="output/S2_groups_acc3/20261010223/checkpoint_best.pth"
test -f "$S2" || { echo "[D1] S2 pretrained checkpoint missing: $S2" >&2; exit 1; }

python -m unittest tests.test_D1_decision_injection -v

mkdir -p logs out
STAMP="$(date +%Y%m%d_%H%M%S)"

run_one() {
  local LETTER="$1"
  local TAG CFG
  if [[ "$LETTER" == "A" ]]; then
    TAG="A_ID_V_ONLY"
  else
    TAG="B_DECISION_Q_ID_V"
  fi
  CFG="tran_configs/D1_${TAG}.yaml"
  local OUT="output/D1_${TAG}_${STAMP}"
  local PRE_LOG="logs/D1_${TAG}_preflight_${STAMP}.log"
  local TR_LOG="logs/D1_${TAG}_train_${STAMP}.log"
  local TS_LOG="logs/D1_${TAG}_test_${STAMP}.log"

  echo "[D1] ${TAG}: Valid preflight from S2 champion (no training)"
  # For D1-A and zero-init D1-B, this should reproduce the S2 validation AUC
  # because final-layer ID-Q has no causal path to the final-prompt logit.
  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    run.evaluate=true run.test_splits=[valid] \
    run.output_dir="output/D1_${TAG}_preflight_${STAMP}" \
    2>&1 | tee "$PRE_LOG"

  python - "$PRE_LOG" <<'PY'
import re
import sys
from pathlib import Path
s=Path(sys.argv[1]).read_text(encoding="utf-8",errors="replace")
v=re.findall(r"rank_0 auc:\s*([0-9]+(?:\.[0-9]+)?)",s)
if not v:
    raise SystemExit("[D1] Could not parse checkpoint Valid AUC; stop before training")
auc=float(v[-1])
print("[D1] PRE-FLIGHT VALID AUC:",auc,flush=True)
if abs(auc-0.7328835632321707)>0.003:
    raise SystemExit(
        "[D1] S2 checkpoint equivalence did not hold. "
        "Do not train; inspect checkpoint and D1 injection mapping."
    )
PY

  echo "[D1] ${TAG}: foreground 3-epoch LLM-only fine-tuning"
  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    run.evaluate=false run.output_dir="$OUT" \
    2>&1 | tee "$TR_LOG"

  local BEST
  BEST="$(find "$OUT" -type f -name checkpoint_best.pth -print -quit)"
  test -n "$BEST" || { echo "[D1] best checkpoint missing" >&2; exit 1; }

  echo "[D1] ${TAG}: test best-Valid checkpoint $BEST"
  python -u train_collm_mf_din.py --cfg-path "$CFG" --options \
    model.ckpt="$(realpath "$BEST")" run.evaluate=true run.test_splits=[test] \
    run.output_dir="output/D1_${TAG}_test_${STAMP}" \
    2>&1 | tee "$TS_LOG"

  printf 'experiment=%s\ncheckpoint=%s\npreflight_log=%s\ntrain_log=%s\ntest_log=%s\n' \
    "$TAG" "$BEST" "$PRE_LOG" "$TR_LOG" "$TS_LOG" \
    > "out/D1_${TAG}_${STAMP}.txt"
  echo "[D1] ${TAG} completed: out/D1_${TAG}_${STAMP}.txt"
}

if [[ "$MODE" == "both" ]]; then
  run_one A
  run_one B
else
  run_one "$MODE"
fi
echo "[D1] all requested experiments finished."
