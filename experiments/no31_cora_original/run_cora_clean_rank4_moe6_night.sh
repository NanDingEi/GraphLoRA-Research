#!/usr/bin/env bash
# Six isolated and serial clean CoRA 4xRank4 MoE runs; no RR1/D4/C6.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
# Compatibility with older GraphLoRA-based RunnerBase versions, capped to 2.
export S2_NIGHT_EPOCHS=2

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
  protocol="CORA_CLEAN_R4_20261111_$tag"
  echo "========== CLEAN R4 START $tag route=$mode strength=$strength =========="
  python -u "$SCRIPT" --run \
    --protocol "$protocol" \
    --route-mode "$mode" \
    --route-strength "$strength" \
    --warmup-microsteps 200 \
    --lr 0.0003 \
    --epochs 2 --patience 2 \
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
    log = here / "logs" / ("CORA_CLEAN_R4_20261111_" + tag) / "CLEAN_R4_MOE.log"
    pairs = pattern.findall(log.read_text(errors="replace")) if log.is_file() else []
    if not pairs:
        rows.append(dict(id=tag, valid_auc="", valid_uauc="", status="NO_METRICS", log=str(log)))
        continue
    best = max((float(auc), float(uauc)) for auc, uauc in pairs)
    rows.append(dict(id=tag, valid_auc=f"{best[0]:.9f}",
                     valid_uauc=f"{best[1]:.9f}", status="PARSED", log=str(log)))
out = here / "out" / "CORA_CLEAN_RANK4_MOE6_20261111_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("[CLEAN R4 SUMMARY]", out, flush=True)
for row in rows:
    print(row["id"], row["valid_auc"], row["valid_uauc"], row["status"])
PY
