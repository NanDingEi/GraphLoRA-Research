#!/usr/bin/env bash
# Six serial, validation-only CoRA full-information MoE experiments.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
# Compatibility with historical GraphLoRA runner variants; no effect in official runner.
export S2_NIGHT_EPOCHS=2
SCRIPT="experiments/no31_cora_original/cora_fullinfo_moe6.py"
[[ -f "$SCRIPT" ]] || { echo "[FATAL] Missing $SCRIPT" >&2; exit 2; }
python "$SCRIPT" --cpu-test

RUNS=(
  'F1_UNIFORM uniform 0.2'
  'F2_USER_L02 user 0.2'
  'F3_PAIR_L02 pair 0.2'
  'F4_LAYER_PAIR_L02 layer_pair 0.2'
  'F5_PAIR_L10 pair 1.0'
  'F6_LAYER_PAIR_L10 layer_pair 1.0'
)

for trial in "${RUNS[@]}"; do
  read -r tag mode strength <<< "$trial"
  protocol="CORA_FULLINFO6_20261110_${tag}"
  echo "========== [START] ${tag} mode=${mode} lambda=${strength} =========="
  python -u "$SCRIPT" \
    --run \
    --protocol "$protocol" \
    --route-mode "$mode" \
    --route-strength "$strength" \
    --balance-weight 0.001 \
    --warmup-microsteps 200 \
    --lr 0.0001 \
    --epochs 2 \
    --patience 2 \
    --min-free-mib 22000
  echo "========== [DONE] ${tag} =========="
done

python - "$HERE" <<'PY'
import csv, pathlib, re, sys
here = pathlib.Path(sys.argv[1])
name='FULLINFO4_RR1_FROZEN'
ids=['F1_UNIFORM','F2_USER_L02','F3_PAIR_L02','F4_LAYER_PAIR_L02','F5_PAIR_L10','F6_LAYER_PAIR_L10']
pattern = re.compile(r'\*\*\*auc:\s*([\d.]+)\s+\*\*\*uauc:\s*\(?([\d.]+)', re.I)
rows=[]
for id in ids:
    protocol=f'CORA_FULLINFO6_20261110_{id}'
    log=here/'logs'/protocol/f'{name}.log'
    pairs=pattern.findall(log.read_text(errors='replace')) if log.exists() else []
    if not pairs:
        rows.append(dict(id=id,valid_auc='',valid_uauc='',delta_auc_vs_rr1='',delta_uauc_vs_rr1='',log=str(log),status='CHECK_LOG'))
        continue
    auc,uauc=max(((float(a),float(u)) for a,u in pairs),key=lambda x:x[0])
    rows.append(dict(id=id,valid_auc=f'{auc:.9f}',valid_uauc=f'{uauc:.9f}',delta_auc_vs_rr1=f'{auc-0.732387:+.9f}',delta_uauc_vs_rr1=f'{uauc-0.701999:+.9f}',log=str(log),status='PARSED'))
out=here/'out'/'CORA_FULLINFO6_20261110_summary.csv'
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('[FULL6 SUMMARY]',out)
for row in rows: print(row['id'],row['valid_auc'],row['valid_uauc'],row['status'])
PY
