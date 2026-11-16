#!/usr/bin/env bash
# Six uncapped, foreground serial clean CoRA 4xRank4 MoE runs; Early Stop patience=5.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
# Runtime patch in cora_clean_rank4_moe6.py removes the runner's forced 1/2-epoch cap.
unset S2_NIGHT_EPOCHS

SCRIPT="experiments/no31_cora_original/cora_clean_rank4_moe6.py"
test -f "$SCRIPT"
python "$SCRIPT" --cpu-test

for row in \
  "F1_UNIFORM uniform 0.2" \
  "F2_USER_L02 user 0.2" \
  "F3_PAIR_L02 pair 0.2" \
  "F4_LAYER_PAIR_L02 layer_pair 0.2" \
  "F5_PAIR_L10 pair 1.0" \
  "F6_LAYER_PAIR_L10 layer_pair 1.0"; do
  read -r tag mode strength <<< "$row"
  protocol="CORA_CLEAN_R4_EARLY5_FIX1_20261111_$tag"
  echo "========== CLEAN R4 START $tag route=$mode strength=$strength =========="
  python -u "$SCRIPT" --run \
    --protocol "$protocol" \
    --route-mode "$mode" \
    --route-strength "$strength" \
    --warmup-microsteps 200 \
    --lr 0.0003 \
    --epochs 0 --patience 5 \
    --min-free-mib 22000
  echo "========== CLEAN R4 DONE $tag =========="
done

python - "$HERE" <<'PY'
import csv
import pathlib
import re
import sys

here = pathlib.Path(sys.argv[1])
tags = ("F1_UNIFORM", "F2_USER_L02", "F3_PAIR_L02",
        "F4_LAYER_PAIR_L02", "F5_PAIR_L10", "F6_LAYER_PAIR_L10")
pattern = re.compile(r"\*\*\*auc:\s*([0-9.]+)\s+\*\*\*uauc:\s*\(?([0-9.]+)", re.I)
rows = []
for tag in tags:
    log = here / "logs" / ("CORA_CLEAN_R4_EARLY5_FIX1_20261111_" + tag) / "CLEAN_R4_MOE.log"
    pairs = pattern.findall(log.read_text(errors="replace")) if log.is_file() else []
    if not pairs:
        rows.append(dict(id=tag, best_epoch="", total_epochs="", valid_auc="", valid_uauc="", status="NO_METRICS", log=str(log)))
        continue
    values = [(float(auc), float(uauc)) for auc, uauc in pairs]
    best_epoch = max(range(len(values)), key=lambda i: values[i][0])
    best = values[best_epoch]
    completed = "[JOINT8 EARLY STOP]" in log.read_text(errors="replace")
    rows.append(dict(id=tag, best_epoch=best_epoch, total_epochs=len(values),
                     valid_auc=f"{best[0]:.9f}", valid_uauc=f"{best[1]:.9f}",
                     status="EARLY_STOP_5" if completed else "CHECK_STOP_REASON", log=str(log)))
out = here / "out" / "CORA_CLEAN_RANK4_MOE6_EARLY5_FIX1_20261111_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("[CLEAN R4 SUMMARY]", out, flush=True)
for row in rows:
    print(row["id"], f"best_epoch={row['best_epoch']}", f"total_epochs={row['total_epochs']}", row["valid_auc"], row["valid_uauc"], row["status"])
PY
