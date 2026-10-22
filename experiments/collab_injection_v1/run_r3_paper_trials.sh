#!/usr/bin/env bash
# ReLLaX CFLoRA and CoRA-inspired L31 trials, strictly foreground GPU0.
# Nothing runs in the background. Fail fast on all CPU/CUDA gates.
set -euo pipefail
cd /home/bbx/user_lx/GraphLoRA_collab_v1
export CUDA_VISIBLE_DEVICES=0
if [[ "${CONDA_DEFAULT_ENV:-}" != 'minigpt4' ]]; then
  echo '[R3-STOP] conda activate minigpt4 first' >&2
  exit 2
fi
mkdir -p logs/collab_v1
echo '[R3-PREFLIGHT] Python syntax, CPU rank logic and matching P2B reference'
python -m py_compile \
  experiments/collab_injection_v1/r3_paper_adapters.py \
  experiments/collab_injection_v1/r3_entry.py \
  experiments/collab_injection_v1/r3_probe.py \
  experiments/collab_injection_v1/r3_train.py \
  experiments/collab_injection_v1/test_r3_cpu.py
python -u experiments/collab_injection_v1/test_r3_cpu.py
python -u experiments/collab_injection_v1/r3_train.py \
  --method cflora --check-source-only "${@}"
for method in cflora cora; do
  cap="${method^^}"
  anchor="logs/collab_v1/R3_${cap}_ANCHOR.json"
  if [[ ! -f "$anchor" ]]; then
    echo "[R3-${cap}-START] 32-sample zero-initialization S2 anchor"
    python -u experiments/collab_injection_v1/r3_probe.py \
      --method "$method" 2>&1 | tee "logs/collab_v1/R3_${cap}_ANCHOR.log"
  fi
  run_dir="logs/collab_v1/R3_COMPARE_20261022/R3_${cap}"
  if [[ -d "$run_dir" ]]; then
    python - "$run_dir" <<'PY'
import json, sys
from pathlib import Path
run=Path(sys.argv[1])
p=run/'manifest.json'
if not p.is_file():
    raise SystemExit('[R3-STOP] existing run has no manifest; do not overwrite')
j=json.loads(p.read_text())
if j.get('status')!='complete':
    raise SystemExit(f"[R3-STOP] partial/incomplete experiment {run}, status={j.get('status')}")
ck=Path(j.get('best_checkpoint',''))
if not ck.is_file():
    raise SystemExit(f'[R3-STOP] completed manifest lacks checkpoint {ck}')
print(f'[R3-RESUME-SKIP] {run} already complete')
PY
    continue
  fi
  echo "[R3-${cap}-START] matched 24-step smoke then full one-epoch Valid"
  python -u experiments/collab_injection_v1/r3_train.py \
    --method "$method" "${@}" \
    2>&1 | tee "logs/collab_v1/R3_${cap}_DRIVER.log"
done
echo '[R3-COMPLETE] Both independent paper-inspired trials finished; no Test run'
