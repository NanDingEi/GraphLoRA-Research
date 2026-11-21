#!/usr/bin/env bash
# Test-evaluate the top-2 valid-AUC HP trials that don't have a test report
# yet. Skips trials already tested; safe to re-run as the queue advances.
set -Eeuo pipefail
EXP="/home/bbx/user_lx/CoRA_official/experiments/no31_cora_original"
SUMMARY="$EXP/out/CORA_HIER_FULLSTACK_20261113/hier_fullstack_valid_summary.csv"
PY=/home/bbx/anaconda3/envs/minigpt4/bin/python
OUT_PROTO="CORA_HP_BEST_TEST_20261114"

TOP2=$("$PY" - "$SUMMARY" "$EXP/out/$OUT_PROTO" <<'PY'
import csv, sys
from pathlib import Path
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
done = Path(sys.argv[2])
scored = sorted(
    ((float(r["valid_auc"]), r["trial"]) for r in rows
     if r.get("valid_auc") and r.get("status") == "EARLY_STOP"),
    reverse=True)
picked = []
for auc, name in scored:
    if (done / f"{name}_test_metrics.csv").exists():
        continue
    picked.append(name)
    if len(picked) == 2:
        break
print("\n".join(picked))
PY
)
echo "[TOP2] selected: $TOP2"
for t in $TOP2; do
  echo "[TOP2] testing $t"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u "$EXP/cora_hp_best_test.py" --run --trial "$t" || \
    echo "[TOP2][WARN] $t test failed; continuing"
done
