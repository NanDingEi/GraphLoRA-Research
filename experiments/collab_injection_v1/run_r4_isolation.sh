#!/usr/bin/env bash
# R4-01/02/03 one after another on GPU0, foreground, fail-fast.
# Verified completed arms are skipped; partial runs block reruns.
set -euo pipefail
cd /home/bbx/user_lx/GraphLoRA_collab_v1
export CUDA_VISIBLE_DEVICES=0
if [[ "${CONDA_DEFAULT_ENV:-}" != 'minigpt4' ]]; then
  echo '[R4-STOP] conda activate minigpt4 first' >&2
  exit 2
fi
mkdir -p logs/collab_v1
python -m py_compile \
 experiments/collab_injection_v1/r4_freeze.py \
 experiments/collab_injection_v1/r4_entry.py \
 experiments/collab_injection_v1/r4_probe.py \
 experiments/collab_injection_v1/r4_train.py \
 experiments/collab_injection_v1/test_r4_cpu.py
python -u experiments/collab_injection_v1/test_r4_cpu.py
python -u experiments/collab_injection_v1/r4_train.py \
  --round 01 --check-source-only "${@}"

for round in 01 02 03; do
  echo "[R4-${round}] checking independent S2 zero-init frozen-model anchor"
  anchor="logs/collab_v1/R4_${round}_FROZEN_ANCHOR.json"
  if [[ -f "$anchor" ]]; then
    python - "$anchor" "$round" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); rid=sys.argv[2]; j=json.loads(p.read_text())
if j.get('status')!='complete' or j.get('round')!=rid or j.get('samples')!=32:
    raise SystemExit(f'[R4-STOP] invalid previous anchor {p}')
print(f'[R4-{rid}-ANCHOR-SKIP] verified {p}')
PY
  else
    python -u experiments/collab_injection_v1/r4_probe.py --round "$round" \
      2>&1 | tee "logs/collab_v1/R4_${round}_ANCHOR.log"
  fi
  case "$round" in
    01) tag='R4_01_CFLORA_ONLY' ;;
    02) tag='R4_02_CORA_ONLY' ;;
    03) tag='R4_03_CORA_V31B' ;;
  esac
  folder="logs/collab_v1/R4_ISOLATION_20261022/$tag"
  if [[ -d "$folder" ]]; then
    python - "$folder" "$round" <<'PY'
from pathlib import Path
import json,sys
folder=Path(sys.argv[1]);rid=sys.argv[2];p=folder/'manifest.json'
if not p.is_file():
    raise SystemExit(f'[R4-STOP] incomplete run folder {folder}')
data=json.loads(p.read_text())
if data.get('status')!='complete' or data.get('round')!=rid:
    raise SystemExit(f"[R4-STOP] partial/failed {folder}, status={data.get('status')}")
checkpoint=Path(data.get('best_checkpoint',''))
if not checkpoint.is_file():
    raise SystemExit(f'[R4-STOP] missing best Valid checkpoint {checkpoint}')
print(f'[R4-{rid}-SKIP] Verified complete {checkpoint}')
PY
    continue
  fi
  echo "[R4-${round}-START] frozen-S2 24-step smoke and full one-epoch Valid"
  python -u experiments/collab_injection_v1/r4_train.py \
    --round "$round" "${@}" \
    2>&1 | tee "logs/collab_v1/R4_${round}_DRIVER.log"
done
echo '[R4-COMPLETE] 3 independent frozen-S2 trials completed. NO TEST SET.'
