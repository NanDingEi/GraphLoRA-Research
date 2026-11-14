#!/usr/bin/env bash
# Pick the best valid-AUC HP trial from the summary CSV and test-eval it
# once (skips if its test report already exists).
set -Eeuo pipefail
EXP="/home/bbx/user_lx/CoRA_official/experiments/no31_cora_original"
SUMMARY="$EXP/out/CORA_HIER_FULLSTACK_20261113/hier_fullstack_valid_summary.csv"
PY=/home/bbx/anaconda3/envs/minigpt4/bin/python
OUT_PROTO="CORA_HP_BEST_TEST_20261114"

TRIAL=$("$PY" - "$SUMMARY" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
scored = [(float(r["valid_auc"]), r["trial"]) for r in rows
          if r.get("valid_auc") and r.get("status") == "EARLY_STOP"]
if not scored:
    raise SystemExit("no scored trials")
print(max(scored)[1])
PY
)
echo "[BETA-WINNER] selected $TRIAL"
if [ -f "$EXP/out/$OUT_PROTO/${TRIAL}_test_metrics.csv" ]; then
  echo "[BETA-WINNER] $TRIAL already tested; skip"
  exit 0
fi
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$EXP/cora_hp_best_test.py" --run --trial "$TRIAL"
