#!/usr/bin/env bash
# R5 01/02/03: resume R4 checkpoints for precisely one additional epoch each.
# Run from conda minigpt4; only GPU0, serial foreground; no test split.
set -Eeuo pipefail
cd /home/bbx/user_lx/GraphLoRA_collab_v1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
mkdir -p logs/collab_v1/R5_EXTRA_EPOCH out/R5_EXTRA_EPOCH

if [[ "${CONDA_DEFAULT_ENV:-}" != "minigpt4" ]]; then
  echo "[R5-STOP] Activate conda minigpt4 first" >&2
  exit 2
fi
python -m py_compile \
  r5_extend_one_epoch.py \
  experiments/collab_injection_v1/r5_r4_runtime_entry.py \
  experiments/collab_injection_v1/r3_paper_adapters.py \
  experiments/collab_injection_v1/r4_freeze.py
python -u experiments/collab_injection_v1/test_r4_cpu.py

for round in 01 02 03; do
  logfile="logs/collab_v1/R5_EXTRA_EPOCH/R5_${round}.log"
  if [[ -e "$logfile" ]]; then
    echo "[R5-STOP] Existing log would be overwritten: $logfile" >&2
    exit 2
  fi
  echo "[R5-${round}-START] $(date -Is)"
  python -u experiments/collab_injection_v1/r5_r4_runtime_entry.py \
    --round "$round" 2>&1 | tee "$logfile"
  grep -q '\[R5-R4-RUNTIME-VERIFIED\]' "$logfile" ||
    { echo "[R5-STOP] $round adapter did not execute" >&2; exit 3; }
  grep -q '\[R5-FROZEN-HASH-OK\]' "$logfile" ||
    { echo "[R5-STOP] $round frozen-weight check missing" >&2; exit 3; }
done

python - <<'PY'
import re
from pathlib import Path
base=Path('logs/collab_v1/R5_EXTRA_EPOCH')
old={'01':(0.729172544,0.683971144),
     '02':(0.730184660,0.684257659),
     '03':(0.730200927,0.684261513)}
lines=['round,epoch1_auc,epoch2_auc,epoch1_uauc,epoch2_uauc,delta_auc,delta_uauc']
for name,(first_auc,first_uauc) in old.items():
    body=(base/f'R5_{name}.log').read_text(errors='replace')
    if body.count('[R5-R4-RUNTIME-VERIFIED]')!=1:
        raise RuntimeError(f'R5-{name}: runtime verification not found exactly once')
    pairs=re.findall(r'\*\*\*auc:\s*([0-9.]+)\s*\*\*\*uauc:\s*\(\s*([0-9.]+)',body)
    if len(pairs)!=1:
        raise RuntimeError(f'R5-{name}: expected exactly one Valid AUC/UAUC, got {pairs}')
    auc,uauc=map(float,pairs[0])
    lines.append(f'{name},{first_auc:.9f},{auc:.9f},{first_uauc:.9f},'
                 f'{uauc:.9f},{auc-first_auc:+.9f},{uauc-first_uauc:+.9f}')
summary='\n'.join(lines)+'\n'
Path('out/R5_EXTRA_EPOCH/summary.csv').write_text(summary)
print('[R5-RESULTS]\n'+summary)
PY
echo "[R5-COMPLETE] 3 genuine R4-adapter resumes completed; VALID ONLY"
