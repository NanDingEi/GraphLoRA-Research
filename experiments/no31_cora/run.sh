#!/usr/bin/env bash
# Historical NO31 CTR=.05 KD=.05 with zero-initialized, ID-only CoRA on L31.
# GPU0, serial FOREGROUND. No S2 checkpoint; pretrained MF retained.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
MODE="${1:-}"
EPOCHS="${2:-1}"
LOG="logs/NO31_CTR_KD_CORA_20261023"
EXP="experiments/no31_cora"

usage() {
  echo "Usage: bash $EXP/run.sh {prepare|control|cora|both|summary} [epochs=1|2]" >&2
  exit 2
}
case "$MODE" in prepare|control|cora|both|summary) ;; *) usage ;; esac
case "$EPOCHS" in 1|2) ;; *) usage ;; esac

preflight() {
  python -m py_compile \
    minigpt4/models/minigpt4rec_v2.py \
    minigpt4/runners/runner_base.py \
    "$EXP/runtime.py" "$EXP/prepare.py" "$EXP/train.py"
  python -u "$EXP/test_cpu.py"
}
if [[ "$MODE" == "prepare" ]]; then
  preflight
  python -u "$EXP/prepare.py" --epochs "$EPOCHS"
  echo "[NO31-CORA-PREPARED] No GPU training launched"
  exit 0
fi
if [[ "$MODE" == "summary" ]]; then
  python - "$LOG" <<'PY'
import re,sys
from pathlib import Path
folder=Path(sys.argv[1])
for arm in ('CONTROL','CORA'):
    p=folder/(arm+'.log')
    if not p.exists():
        print(arm,'not run')
        continue
    lines=p.read_text(errors='replace').splitlines()
    scores=[]
    for row in lines:
        if '***auc:' in row and '***uauc:' in row:
            a=re.search(r'\*\*\*auc:\s*([0-9.]+)',row)
            u=re.search(r'\*\*\*uauc:\s*\(?([0-9.]+)',row)
            if a and u:scores.append((float(a.group(1)),float(u.group(1))))
    print(arm,'Valid AUC/UAUC by epoch:',scores,
          'best AUC:',max((x[0] for x in scores),default=None))
PY
  exit 0
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "minigpt4" ]]; then
  echo "[NO31-CORA-STOP] Activate conda minigpt4 first" >&2
  exit 2
fi
preflight
test -f "$LOG/CORA.yaml" || {
  echo "[NO31-CORA-STOP] Run: bash $EXP/run.sh prepare $EPOCHS" >&2; exit 2; }
python - "$LOG/CORA.yaml" "$EPOCHS" <<'PY'
import yaml,sys
cfg=yaml.safe_load(open(sys.argv[1]))
assert cfg['run']['no31_stop_after_epochs']==int(sys.argv[2]),(
    'Requested epochs differ from prepared config; do not overwrite original results')
PY

run_arm() {
  local arm="$1" mode log yaml
  if [[ "$arm" == "CONTROL" ]]; then mode=control; else mode=cora; fi
  log="$LOG/$arm.log"; yaml="$LOG/$arm.yaml"
  test -f "$yaml" || { echo "[NO31-CORA-STOP] $yaml missing" >&2; exit 2; }
  test ! -e "$log" || {
    echo "[NO31-CORA-STOP] Existing $log, refusing to overwrite. Use summary." >&2; exit 2; }
  echo "[NO31-CORA-START] mode=$mode epochs=$EPOCHS seed=42 GPU0 $(date -Is)"
  python -u "$EXP/train.py" --cfg-path "$yaml" --mode "$mode" \
    --cora-gain 0.1 2>&1 | tee "$log" | grep --line-buffered -E \
      '\[NO31-|\[NIGHT8-|Averaged stats:|Saving checkpoint|Traceback|Error:|Train: data epoch:.*11296/'
  python - "$log" "$EPOCHS" "$mode" <<'PY'
import re,sys
from pathlib import Path
p=Path(sys.argv[1]); epochs=int(sys.argv[2]);mode=sys.argv[3]
s=p.read_text(errors='replace')
if f'[NIGHT8-STOP] trained_epochs={epochs}' not in s:
    raise RuntimeError('Incomplete historical NO31 epoch budget')
if '[NIGHT8-MODEL] gnn_scale=0 alpha=0.0' not in s:
    raise RuntimeError('NO31 original L31 injection was not disabled')
if '[NIGHT8-LOSS]' not in s:
    raise RuntimeError('Historical CTR/KD losses did not execute')
if mode=='cora':
    if '[NO31-CORA-COUNTERS]' not in s:
        raise RuntimeError('CoRA missing forward counters')
    import json
    m=re.search(r'\[NO31-CORA-COUNTERS\]\s*(\{[^\n]+\})',s)
    counts=json.loads(m.group(1))
    for k in ('q_cora_forward','v_cora_forward'):
        if counts.get(k,0)<=0:
            raise RuntimeError(f'Inactive CoRA path {k}')
rows=[l for l in s.splitlines() if '***auc:' in l and '***uauc:' in l]
if len(rows)!=epochs:
    raise RuntimeError(f'Expected {epochs} Valid evaluations, got {len(rows)}')
for ep,row in enumerate(rows):
    a=re.search(r'\*\*\*auc:\s*([0-9.]+)',row)
    u=re.search(r'\*\*\*uauc:\s*\(?([0-9.]+)',row)
    assert a and u
    print(f"[NO31-CORA-VALID] mode={mode} epoch={ep} "
          f"AUC={float(a.group(1)):.9f} UAUC={float(u.group(1)):.9f}")
PY
}
case "$MODE" in
  control) run_arm CONTROL ;;
  cora) run_arm CORA ;;
  both) run_arm CONTROL; run_arm CORA ;;
esac
bash "$EXP/run.sh" summary
