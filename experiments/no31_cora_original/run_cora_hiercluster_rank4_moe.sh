#!/usr/bin/env bash
# Six foreground, serial Group+Personalized CoRA-MoE comparisons.
# One personalized + three group experts (flat or truly nested), 4 x Rank4.
# No nohup, no screen, no background launch.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
unset S2_NIGHT_EPOCHS

SCRIPT="experiments/no31_cora_original/cora_hiercluster_rank4_moe.py"
test -f "$SCRIPT"
python "$SCRIPT" --cpu-test

for spec in \
  "B0_RAW_UNIFORM raw uniform 0.2" \
  "B1_FLAT_UNIFORM flat uniform 0.2" \
  "H1_HIER_UNIFORM hier uniform 0.2" \
  "H2_HIER_PAIR_L02 hier pair 0.2" \
  "H3_HIER_PAIR_L10 hier pair 1.0" \
  "H4_HIER_LAYERPAIR_L10 hier layer_pair 1.0"
do
  read -r tag conditioning router strength <<< "$spec"
  protocol="CORA_HCLUSTER_R4_20261112_$tag"
  echo "============ [HCLUSTER START] $tag kind=$conditioning route=$router lambda=$strength ============"
  python -u "$SCRIPT" \
    --run --protocol "$protocol" \
    --condition-kind "$conditioning" \
    --route-mode "$router" \
    --route-strength "$strength" \
    --warmup-microsteps 200 \
    --lr 0.0003 \
    --epochs 0 \
    --patience 5 \
    --min-free-mib 22000
  echo "============ [HCLUSTER DONE] $tag ============"
done

python - "$HERE" <<'PY'
import csv
import pathlib
import re
import sys

here = pathlib.Path(sys.argv[1])
names = (
    "B0_RAW_UNIFORM",
    "B1_FLAT_UNIFORM",
    "H1_HIER_UNIFORM",
    "H2_HIER_PAIR_L02",
    "H3_HIER_PAIR_L10",
    "H4_HIER_LAYERPAIR_L10",
)
pattern = re.compile(r"\*\*\*auc:\s*([0-9.]+)\s+\*\*\*uauc:\s*([0-9.]+)", re.I)
rows = []
for name in names:
    protocol = f"CORA_HCLUSTER_R4_20261112_{name}"
    path = here / "logs" / protocol / "CLEAN_R4_MOE.log"
    data = path.read_text(errors="replace") if path.is_file() else ""
    values = [(float(a), float(u)) for a, u in pattern.findall(data)]
    if values:
        best = max(range(len(values)), key=lambda k: values[k][0])
        row = {
            "trial": name, "best_epoch_0_based": best,
            "evaluated_epochs": len(values),
            "valid_auc": f"{values[best][0]:.9f}",
            "valid_uauc_at_auc_best": f"{values[best][1]:.9f}",
            "early_stop": "[JOINT8 EARLY STOP]" in data,
            "log": str(path),
        }
    else:
        row = {
            "trial": name, "best_epoch_0_based": "",
            "evaluated_epochs": 0, "valid_auc": "", "valid_uauc_at_auc_best": "",
            "early_stop": False, "log": str(path),
        }
    rows.append(row)
out = here / "out" / "CORA_HCLUSTER_RANK4_MOE_20261112_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("[HCLUSTER SUMMARY]", out)
for row in rows:
    print(row["trial"], f"best_ep={row['best_epoch_0_based']}",
          f"epochs={row['evaluated_epochs']}",
          f"AUC={row['valid_auc']}",
          f"UAUC={row['valid_uauc_at_auc_best']}",
          f"early_stop={row['early_stop']}")
PY
