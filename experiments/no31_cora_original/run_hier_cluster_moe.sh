#!/usr/bin/env bash
# Heterogeneous cluster-group + personal expert MoE: two lambda variants.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
unset S2_NIGHT_EPOCHS
PY=/home/bbx/anaconda3/envs/minigpt4/bin/python

SCRIPT="experiments/no31_cora_original/cora_hier_cluster_moe6.py"
test -f "$SCRIPT"
"$PY" "$SCRIPT" --cpu-test

for row in "H1_L02 0.2" "H2_L10 1.0"; do
  read -r tag strength <<< "$row"
  protocol="CORA_HIER_CLUSTER_MOE_20261112_${tag}"
  echo "========== [HCP START] ${tag} lambda=${strength} =========="
  "$PY" -u "$SCRIPT" --run \
    --protocol "$protocol" \
    --route-strength "$strength" \
    --warmup-microsteps 200 \
    --lr 0.0003 \
    --epochs 0 --patience 5 \
    --min-free-mib 22000
  echo "========== [HCP DONE] ${tag} =========="
done

"$PY" - "$HERE" <<'PY'
import csv, pathlib, re, sys
here = pathlib.Path(sys.argv[1])
tags = ("H1_L02", "H2_L10")
pattern = re.compile(r"\*\*\*auc:\s*([0-9.]+)\s+\*\*\*uauc:\s*\(?([0-9.]+)", re.I)
rows = []
for tag in tags:
    log = here / "logs" / ("CORA_HIER_CLUSTER_MOE_20261112_" + tag) / "HIER_MOE.log"
    pairs = pattern.findall(log.read_text(errors="replace")) if log.is_file() else []
    if not pairs:
        rows.append(dict(id=tag, valid_auc="", valid_uauc="", status="NO_METRICS"))
        continue
    values = [(float(a), float(u)) for a, u in pairs]
    best_i = max(range(len(values)), key=lambda i: values[i][0])
    rows.append(dict(id=tag,
                     valid_auc=f"{values[best_i][0]:.9f}",
                     valid_uauc=f"{values[best_i][1]:.9f}",
                     status=f"best_ep={best_i} total={len(values)}"))
out = here / "out" / "CORA_HIER_CLUSTER_MOE_20261112_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("[HCP SUMMARY]", out)
for r in rows:
    print(r)
PY
