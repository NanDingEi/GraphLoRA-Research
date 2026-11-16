#!/usr/bin/env bash
# Foreground, serial ONLY. Three new independent 2Personal+2Group 4xRank4 runs.
# Old six HCLUSTER P5 trial directories, logs and best checkpoints untouched.
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
unset S2_NIGHT_EPOCHS

SCRIPT="experiments/no31_cora_original/cora_mix22_rank4_moe3.py"
test -f "$SCRIPT"

python -m py_compile "$SCRIPT"
python "$SCRIPT" --cpu-test

for trial in \
  O1_RAW2_G2_UNIFORM \
  O2_RESID_G2_UNIFORM \
  O3_RESID_G2_PAIR02
do
  echo "========== [2P2G START] $trial =========="
  python -u "$SCRIPT" --run --trial "$trial" --min-free-mib 22000
  echo "========== [2P2G DONE] $trial =========="
done

python - "$HERE" <<'PY'
import csv
import pathlib
import re
import sys

here = pathlib.Path(sys.argv[1])
names = ("O1_RAW2_G2_UNIFORM", "O2_RESID_G2_UNIFORM", "O3_RESID_G2_PAIR02")
pat = re.compile(r"\*\*\*auc:\s*([0-9.]+)\s+\*\*\*uauc:\s*([0-9.]+)", re.I)
rows = []
for name in names:
    protocol = "CORA_2P2G_R4_20261112_" + name
    log = here / "logs" / protocol / "CLEAN_R4_MOE.log"
    content = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    values = [(float(a), float(u)) for a, u in pat.findall(content)]
    if values:
        ep = max(range(len(values)), key=lambda i: values[i][0])
        rows.append({
            "trial": name,
            "best_epoch_0based": ep,
            "valid_auc": f"{values[ep][0]:.9f}",
            "valid_uauc_at_best_auc": f"{values[ep][1]:.9f}",
            "evaluated_epochs": len(values),
            "early_stop_10": "[JOINT8 EARLY STOP] 10 consecutive epochs" in content,
            "log": str(log),
        })
    else:
        rows.append({
            "trial": name,
            "best_epoch_0based": "",
            "valid_auc": "",
            "valid_uauc_at_best_auc": "",
            "evaluated_epochs": 0,
            "early_stop_10": False,
            "log": str(log),
        })
out = here / "out" / "CORA_2P2G_R4_20261112_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("[2P2G SUMMARY] " + str(out), flush=True)
for row in rows:
    print(row["trial"], "best_ep=", row["best_epoch_0based"],
          "valid_auc=", row["valid_auc"],
          "valid_uauc=", row["valid_uauc_at_best_auc"],
          "n_epochs=", row["evaluated_epochs"],
          "early_stop=", row["early_stop_10"], flush=True)
PY
