#!/usr/bin/env bash
# S2 rollback + minimal CoRA. Runs on GPU0 in foreground, no background jobs.
# Phases: prepare (CPU), probe (GPU model initialisation ONLY),
# baseline (3 full original-S2 epochs), cora (3 full epochs), all.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
ROOT="$(pwd)"
EXP="experiments/s2_cora"
LOG="logs/S2_ROLLBACK_CORA_20261022"
CONFIG="tran_configs/S2_groups_acc3_recovered.yaml"

usage() {
  echo "Usage: bash $EXP/run.sh {prepare|probe|baseline|cora|all}"
  exit 2
}
PHASE="${1:-}"; [[ -n "$PHASE" ]] || usage

preflight() {
  test -f "$CONFIG" || { echo "[S2-FAIL] S2 historical YAML missing" >&2; exit 2; }
  test -f "$EXP/s2_entry.py" || { echo "[S2-FAIL] S2 runner missing" >&2; exit 2; }
  python -m py_compile minigpt4/models/minigpt4rec_v2.py \
    "$EXP/s2_cora_adapter.py" "$EXP/s2_entry.py" \
    "$EXP/prepare_s2_cora.py"
  python -u "$EXP/test_s2_cora_cpu.py"
}

prepare() {
  preflight
  python -u "$EXP/prepare_s2_cora.py"
  echo "[S2-PREPARE-OK] S2 original runtime and identical YAMLs ready"
}

probe() {
  test -f "$LOG/BASELINE.yaml" || { echo "[S2-FAIL] Run prepare first" >&2; exit 2; }
  test -f "$LOG/CORA.yaml" || { echo "[S2-FAIL] Run prepare first" >&2; exit 2; }
  for mode in baseline cora; do
    arm="${mode^^}"
    python -u "$EXP/s2_entry.py" --mode "$mode" \
      --cfg-path "$LOG/$arm.yaml" --fingerprint "$LOG/${arm}_probe.json" \
      --probe-only 2>&1 | tee "$LOG/${arm}_probe.log"
  done
  python - "$LOG" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
a=json.loads((p/'BASELINE_probe.json').read_text())['groups']
b=json.loads((p/'CORA_probe.json').read_text())['groups']
wrong=[k for k in ('MF','GNN','Projector','LoRA')
       if a[k]['sha256']!=b[k]['sha256']]
if wrong:
    raise RuntimeError(f"Unfair initial shared model parameters: {wrong}")
print("[S2-INITIAL-PARITY-PASS] MF + GNN + Projector + LoRA exact SHA256")
PY
}

verify_parity() {
  python - "$LOG" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
a=json.loads((p/'BASELINE_probe.json').read_text())['groups']
b=json.loads((p/'CORA_probe.json').read_text())['groups']
for x in ('MF','GNN','Projector','LoRA'):
    assert a[x]['sha256']==b[x]['sha256'],x
print("[S2-INITIAL-PARITY-PASS]")
PY
}

extract_auc() {
  python - "$1" <<'PY'
import re,sys
from pathlib import Path
s=Path(sys.argv[1]).read_text(errors='replace')
a=[float(x) for x in re.findall(r'\*\*\*auc:\s*([0-9.eE+-]+)',s)]
u=[float(x) for x in re.findall(r'\*\*\*uauc:\(\s*([0-9.eE+-]+)',s)]
if len(a)!=3 or len(u)!=3:
    raise RuntimeError(f"Expected 3 completed Valid epochs. AUC={a}, UAUC={u}")
for i,(auc,uauc) in enumerate(zip(a,u)):
    print(f'[S2-VALID] epoch={i} AUC={auc:.9f} UAUC={uauc:.9f}')
print(f'[S2-VALID-BEST] AUC={max(a):.9f}')
if Path(sys.argv[1]).name == 'BASELINE_train.log' and max(a) < 0.72:
    raise RuntimeError(f'BASELINE reproduced only {max(a):.6f} (historical 0.732884); stop BEFORE CoRA')
PY
}

train_arm() {
  local arm="$1" mode="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  local logfile="$LOG/${arm}_train.log"
  test -f "$LOG/${arm}.yaml" || { echo "[S2-FAIL] Run prepare first" >&2; exit 2; }
  test ! -e "$logfile" || { echo "[S2-FAIL] Existing log $logfile; refusing overwrite" >&2; exit 2; }
  echo "[S2-TRAIN] arm=$arm epochs=3 GPU0 $(date -Is)"
  python -u "$EXP/s2_entry.py" \
    --cfg-path "$LOG/${arm}.yaml" --mode "$mode" \
    --fingerprint "$LOG/${arm}_train_init.json" \
    2>&1 | tee "$logfile" | grep --line-buffered -E \
      'S2-|INFO|Train: data epoch:.*(11296/|\[    0/)|Averaged stats:|Traceback|Error:'
  extract_auc "$logfile"
  python - "$LOG" "$arm" <<'PY'
import json,sys
from pathlib import Path
base,arm=Path(sys.argv[1]),sys.argv[2]
a=json.loads((base/(arm+'_probe.json')).read_text())['groups']
b=json.loads((base/(arm+'_train_init.json')).read_text())['groups']
for k in ('MF','GNN','Projector','LoRA'):
    if a[k]['sha256']!=b[k]['sha256']:
        raise RuntimeError(f'{arm}: probe vs train initial weights mismatch: {k}')
print(f'[S2-TRAIN-INIT-VERIFIED] {arm}')
PY
}

baseline() {
  verify_parity
  train_arm BASELINE
}
cora() {
  verify_parity
  if [[ ! -f "$LOG/BASELINE_train.log" ]]; then
    echo "[S2-FAIL] Baseline first: historical reproduction must pass 0.72" >&2
    exit 2
  fi
  extract_auc "$LOG/BASELINE_train.log"
  train_arm CORA
}
case "$PHASE" in
  prepare) prepare;;
  probe) preflight;probe;;
  baseline) preflight;baseline;;
  cora) preflight;cora;;
  all) prepare;probe;baseline;cora;;
  *) usage;;
esac
