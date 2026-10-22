#!/usr/bin/env bash
# R5: probe and then continue the three R4 adapters one extra epoch, serially.
set -Eeuo pipefail
cd /home/bbx/user_lx/GraphLoRA_collab_v1
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-/home/bbx/anaconda3/envs/minigpt4/bin/python}"
mkdir -p logs/collab_v1/R5_EXTRA_EPOCH out/R5_EXTRA_EPOCH
"$PYTHON" -m py_compile r5_r3_bootstrap.py r5_extend_one_epoch.py

# Fail fast if the local model factory does not support the R4 R3 adapter.
for id in 01 02 03; do
  echo "=== R5-${id} R3 adapter construction probe ==="
  "$PYTHON" -u r5_r3_bootstrap.py --round "$id" --probe-only 2>&1 |
    tee "logs/collab_v1/R5_EXTRA_EPOCH/R5_${id}_probe.log"
  grep -q '\[R5-R3-PROBE-PASS\]' "logs/collab_v1/R5_EXTRA_EPOCH/R5_${id}_probe.log"
done

# Full training, one additional epoch per R4 checkpoint.
for id in 01 02 03; do
  echo "=== R5-${id} full extra epoch START $(date -Is) ==="
  "$PYTHON" -u r5_r3_bootstrap.py --round "$id" 2>&1 |
    tee "logs/collab_v1/R5_EXTRA_EPOCH/R5_${id}.log"
  grep -q '\[R5-TRAIN-COMPLETE\]' "logs/collab_v1/R5_EXTRA_EPOCH/R5_${id}.log"
done

"$PYTHON" - <<'PY'
import re
from pathlib import Path
base = Path('logs/collab_v1/R5_EXTRA_EPOCH')
old = {'01': (0.729172544, 0.683971144),
       '02': (0.730184660, 0.684257659),
       '03': (0.730200927, 0.684261513)}
rows = ['round,epoch1_auc,epoch2_auc,epoch1_uauc,epoch2_uauc,delta_auc,delta_uauc']
for key in ('01', '02', '03'):
    lines = (base / f'R5_{key}.log').read_text(errors='replace').splitlines()
    vals = []
    for line in lines:
        if 'Averaged stats:' not in line or 'auc:' not in line:
            continue
        m = re.search(r'\*\*\*auc:\s*([0-9.]+).*?\*\*\*uauc:\s*\(\s*([0-9.]+)', line)
        if m:
            vals.append((float(m.group(1)), float(m.group(2))))
    if len(vals) != 1:
        raise RuntimeError(f'R5-{key}: expected one validation result, got {len(vals)}')
    auc, uauc = vals[0]
    old_auc, old_uauc = old[key]
    rows.append(f'{key},{old_auc:.9f},{auc:.9f},{old_uauc:.9f},{uauc:.9f},{auc-old_auc:+.9f},{uauc-old_uauc:+.9f}')
summary = '\n'.join(rows) + '\n'
Path('out/R5_EXTRA_EPOCH/summary.csv').write_text(summary)
print(summary)
PY
